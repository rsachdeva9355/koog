package ai.koog.prompt.executor.clients.openai.realtime

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.executor.clients.LLMClientException
import ai.koog.prompt.executor.clients.openai.base.OpenAICompatibleToolDescriptorSchemaGenerator
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import ai.koog.prompt.streaming.StreamFrame
import ai.koog.prompt.streaming.requireEndFrame
import ai.koog.prompt.streaming.toMessageResponses
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.client.plugins.websocket.wss
import io.ktor.http.HttpHeaders
import io.ktor.websocket.Frame
import kotlinx.coroutines.channels.ClosedReceiveChannelException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.toList
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.parseToJsonElement
import org.jetbrains.annotations.ApiStatus.Experimental
import kotlin.time.Clock

private val logger = KotlinLogging.logger {}

/**
 * LLM client that connects to the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
 * via a WebSocket connection, supporting text responses and function/tool calls.
 *
 * Each call to [execute] or [executeStreaming] opens a dedicated WebSocket session, sends the
 * conversation items converted from the [Prompt], triggers a response, and streams the resulting
 * [StreamFrame]s back to the caller. The connection is closed automatically when the response
 * is complete.
 *
 * This client implements [LLMClient] and is therefore fully compatible with the koog agent
 * framework — it can be used wherever a `PromptExecutor` is accepted.
 *
 * ### Supported features
 * - Text responses streamed as [StreamFrame.TextDelta] / [StreamFrame.TextComplete]
 * - Function/tool calls streamed as [StreamFrame.ToolCallDelta] / [StreamFrame.ToolCallComplete]
 * - System instructions via [Message.System] (mapped to Realtime session `instructions`)
 * - Multi-turn conversations (each `execute` call opens its own WebSocket session)
 *
 * ### Not supported
 * - Audio input/output — use the OpenAI Audio models with `OpenAILLMClient` instead
 * - Content moderation — use `OpenAILLMClient.moderate()` instead
 *
 * ### Usage
 * ```kotlin
 * val client = OpenAIRealtimeClient(apiKey = System.getenv("OPENAI_API_KEY"))
 *
 * val executor = SimplePromptExecutorStrategy(client)
 * val agent = AIAgent(promptExecutor = executor, ...) { ... }
 * ```
 *
 * @param apiKey OpenAI API key (e.g. from the `OPENAI_API_KEY` environment variable).
 * @param settings Connection and behaviour configuration. See [OpenAIRealtimeClientSettings].
 * @param baseClient Ktor [HttpClient] to use. Must have [WebSockets] installed; if you supply a
 *   custom client ensure the engine supports `wss://` (TLS WebSockets).
 * @param clock Clock used to populate [ResponseMetaInfo] timestamps.
 * @param toolsConverter Schema generator used to build JSON parameter schemas for tools.
 */
@Experimental
public class OpenAIRealtimeClient @JvmOverloads constructor(
    private val apiKey: String,
    private val settings: OpenAIRealtimeClientSettings = OpenAIRealtimeClientSettings(),
    baseClient: HttpClient = HttpClient { install(WebSockets) },
    private val clock: Clock = Clock.System,
    private val toolsConverter: OpenAICompatibleToolDescriptorSchemaGenerator =
        OpenAICompatibleToolDescriptorSchemaGenerator(),
) : LLMClient() {

    override val clientName: String get() = "OpenAIRealtimeClient"

    private val httpClient: HttpClient = baseClient

    /**
     * JSON codec for Realtime API events.
     * Uses `"type"` as the class discriminator to match the wire format.
     */
    private val json: Json = Json {
        classDiscriminator = "type"
        ignoreUnknownKeys = true
        encodeDefaults = false
        explicitNulls = false
    }

    // -------------------------------------------------------------------------
    // LLMClientAPI implementation
    // -------------------------------------------------------------------------

    override fun llmProvider(): LLMProvider = LLMProvider.OpenAI

    /**
     * Executes the prompt over a WebSocket session and returns the full response as a list of
     * [Message.Response] objects.
     */
    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): List<Message.Response> =
        executeStreaming(prompt, model, tools).toList().toMessageResponses()

    /**
     * Opens a WebSocket connection to the OpenAI Realtime API, configures the session with the
     * given [tools] and system instructions extracted from [prompt], sends all conversation items,
     * triggers response generation, and returns a cold [Flow] of [StreamFrame]s.
     *
     * The returned flow is guaranteed to end with a [StreamFrame.End] frame (enforced by
     * [requireEndFrame]). Unexpected disconnections result in an [ai.koog.prompt.streaming.IncompleteStreamException].
     *
     * @param prompt Conversation context including messages and LLM parameters.
     * @param model Must have [ai.koog.prompt.llm.LLMCapability.OpenAIEndpoint.Realtime] capability.
     * @param tools Function/tool definitions available to the model for this turn.
     */
    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Flow<StreamFrame> = channelFlow {
        val systemInstructions = prompt.messages
            .filterIsInstance<Message.System>()
            .lastOrNull()?.content

        logger.debug { "Opening Realtime WebSocket session for model=${model.id}" }

        try {
            httpClient.wss(
                host = settings.baseUrl,
                path = "${settings.path}?model=${model.id}",
                request = {
                    headers.append(HttpHeaders.Authorization, "Bearer $apiKey")
                    headers.append("OpenAI-Beta", "realtime=v1")
                },
            ) {
                // ---- 1. Configure session ----------------------------------------
                sendEvent(
                    RealtimeSessionUpdateEvent(
                        session = RealtimeSessionConfig(
                            modalities = listOf("text"),
                            instructions = systemInstructions,
                            tools = tools.map { it.toRealtimeTool() }.takeIf { it.isNotEmpty() },
                            toolChoice = if (tools.isEmpty()) null else JsonPrimitive("auto"),
                            temperature = prompt.params.temperature,
                            turnDetection = null, // disable server-side VAD
                        )
                    )
                )

                // ---- 2. Append conversation items --------------------------------
                for (message in prompt.messages) {
                    val item = message.toRealtimeItem() ?: continue
                    sendEvent(RealtimeConversationItemCreateEvent(item = item))
                }

                // ---- 3. Trigger response generation -----------------------------
                sendEvent(RealtimeResponseCreateEvent())

                // ---- 4. Stream response events ----------------------------------
                // Maps call_id -> function name, populated when response.output_item.added arrives.
                val functionCallNames = mutableMapOf<String, String>()

                try {
                    for (frame in incoming) {
                        if (frame !is Frame.Text) continue

                        val event = decodeServerEvent(frame.readText()) ?: continue

                        when (event) {
                            is RealtimeErrorEvent ->
                                throw LLMClientException(
                                    clientName = clientName,
                                    message = buildString {
                                        append("OpenAI Realtime API error")
                                        val code = event.error.code ?: event.error.type
                                        if (code != null) append(" ($code)")
                                        append(": ")
                                        append(event.error.message)
                                    },
                                )

                            is RealtimeResponseOutputItemAddedEvent -> {
                                val item = event.item
                                if (item.type == "function_call" &&
                                    item.callId != null &&
                                    item.name != null
                                ) {
                                    functionCallNames[item.callId] = item.name
                                }
                            }

                            is RealtimeResponseTextDeltaEvent ->
                                send(StreamFrame.TextDelta(event.delta, event.outputIndex))

                            is RealtimeResponseTextDoneEvent ->
                                send(StreamFrame.TextComplete(event.text, event.outputIndex))

                            is RealtimeResponseFunctionCallArgumentsDeltaEvent ->
                                send(
                                    StreamFrame.ToolCallDelta(
                                        id = event.callId,
                                        name = functionCallNames[event.callId],
                                        content = event.delta,
                                        index = event.outputIndex,
                                    )
                                )

                            is RealtimeResponseFunctionCallArgumentsDoneEvent ->
                                send(
                                    StreamFrame.ToolCallComplete(
                                        id = event.callId,
                                        name = event.name,
                                        content = event.arguments,
                                        index = event.outputIndex,
                                    )
                                )

                            is RealtimeResponseDoneEvent -> {
                                val usage = event.response.usage
                                send(
                                    StreamFrame.End(
                                        finishReason = event.response.status,
                                        metaInfo = ResponseMetaInfo.create(
                                            clock = clock,
                                            totalTokensCount = usage?.totalTokens,
                                            inputTokensCount = usage?.inputTokens,
                                            outputTokensCount = usage?.outputTokens,
                                        ),
                                    )
                                )
                                break // exit the receive loop; wss block will close the connection
                            }

                            // session.created, session.updated, etc. — no action needed
                            else -> {}
                        }
                    }
                } catch (e: ClosedReceiveChannelException) {
                    // Server closed the connection before response.done — handled by requireEndFrame()
                    logger.debug { "Realtime WebSocket incoming channel closed unexpectedly" }
                }
            }
        } catch (e: LLMClientException) {
            throw e
        } catch (e: Exception) {
            throw LLMClientException(
                clientName = clientName,
                message = "OpenAI Realtime WebSocket error: ${e.message}",
                cause = e,
            )
        }
    }.requireEndFrame()

    /**
     * Not supported — the Realtime API does not expose a moderation endpoint.
     * Use [ai.koog.prompt.executor.clients.openai.OpenAILLMClient] for moderation.
     */
    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        throw UnsupportedOperationException(
            "Content moderation is not supported by the OpenAI Realtime API. " +
                "Use OpenAILLMClient for moderation."
        )

    override fun close() {
        logger.debug { "Closing $clientName" }
        httpClient.close()
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /** Serialises [event] to JSON and sends it as a WebSocket text frame. */
    private suspend fun io.ktor.websocket.DefaultWebSocketSession.sendEvent(
        event: RealtimeClientEvent,
    ) {
        val text = json.encodeToString(event)
        logger.debug { "→ Realtime: $text" }
        send(Frame.Text(text))
    }

    /**
     * Parses a JSON string from the server into a typed event object.
     * Returns `null` for event types that are not relevant to text-mode operation.
     */
    private fun decodeServerEvent(text: String): Any? {
        logger.debug { "← Realtime: $text" }
        return try {
            val element = json.parseToJsonElement(text)
            val type = element.jsonObject["type"]?.jsonPrimitive?.content ?: return null

            when (type) {
                "session.created" ->
                    json.decodeFromJsonElement<RealtimeSessionCreatedEvent>(element)
                "session.updated" ->
                    json.decodeFromJsonElement<RealtimeSessionUpdatedEvent>(element)
                "response.output_item.added" ->
                    json.decodeFromJsonElement<RealtimeResponseOutputItemAddedEvent>(element)
                "response.text.delta" ->
                    json.decodeFromJsonElement<RealtimeResponseTextDeltaEvent>(element)
                "response.text.done" ->
                    json.decodeFromJsonElement<RealtimeResponseTextDoneEvent>(element)
                "response.function_call_arguments.delta" ->
                    json.decodeFromJsonElement<RealtimeResponseFunctionCallArgumentsDeltaEvent>(element)
                "response.function_call_arguments.done" ->
                    json.decodeFromJsonElement<RealtimeResponseFunctionCallArgumentsDoneEvent>(element)
                "response.done" ->
                    json.decodeFromJsonElement<RealtimeResponseDoneEvent>(element)
                "error" ->
                    json.decodeFromJsonElement<RealtimeErrorEvent>(element)
                else -> null
            }
        } catch (e: Exception) {
            logger.debug(e) { "Failed to decode Realtime server event: $text" }
            null
        }
    }

    /**
     * Converts a koog [Message] to a [RealtimeItem] for the conversation.
     * [Message.System] returns `null` because system instructions are sent via `session.update`,
     * not as conversation items.
     */
    private fun Message.toRealtimeItem(): RealtimeItem? = when (this) {
        is Message.System -> null

        is Message.User -> RealtimeMessageItem(
            role = "user",
            content = listOf(RealtimeInputTextContent(text = content)),
        )

        is Message.Assistant -> RealtimeMessageItem(
            role = "assistant",
            content = listOf(RealtimeTextContent(text = content)),
        )

        is Message.Tool.Call -> RealtimeFunctionCallItem(
            name = tool,
            arguments = content,
            callId = id ?: tool,
        )

        is Message.Tool.Result -> RealtimeFunctionCallOutputItem(
            callId = id ?: tool,
            output = content,
        )

        is Message.Reasoning -> null // Not applicable in text-only Realtime mode
    }

    /** Converts a [ToolDescriptor] to a [RealtimeFunctionTool] for session configuration. */
    private fun ToolDescriptor.toRealtimeTool(): RealtimeFunctionTool = RealtimeFunctionTool(
        name = name,
        description = description,
        parameters = toolsConverter.generate(this),
    )
}

// -------------------------------------------------------------------------
// Server event data classes (deserialized via manual dispatch — not sealed)
// -------------------------------------------------------------------------

@kotlinx.serialization.Serializable
internal data class RealtimeSessionCreatedEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    val session: kotlinx.serialization.json.JsonObject? = null,
)

@kotlinx.serialization.Serializable
internal data class RealtimeSessionUpdatedEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    val session: kotlinx.serialization.json.JsonObject? = null,
)

@kotlinx.serialization.Serializable
internal data class RealtimeResponseOutputItemAddedEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    @kotlinx.serialization.SerialName("response_id") val responseId: String,
    @kotlinx.serialization.SerialName("output_index") val outputIndex: Int,
    val item: RealtimeOutputItemInfo,
)

@kotlinx.serialization.Serializable
internal data class RealtimeResponseTextDeltaEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    @kotlinx.serialization.SerialName("response_id") val responseId: String,
    @kotlinx.serialization.SerialName("item_id") val itemId: String,
    @kotlinx.serialization.SerialName("output_index") val outputIndex: Int,
    @kotlinx.serialization.SerialName("content_index") val contentIndex: Int,
    val delta: String,
)

@kotlinx.serialization.Serializable
internal data class RealtimeResponseTextDoneEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    @kotlinx.serialization.SerialName("response_id") val responseId: String,
    @kotlinx.serialization.SerialName("item_id") val itemId: String,
    @kotlinx.serialization.SerialName("output_index") val outputIndex: Int,
    @kotlinx.serialization.SerialName("content_index") val contentIndex: Int,
    val text: String,
)

@kotlinx.serialization.Serializable
internal data class RealtimeResponseFunctionCallArgumentsDeltaEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    @kotlinx.serialization.SerialName("response_id") val responseId: String,
    @kotlinx.serialization.SerialName("item_id") val itemId: String,
    @kotlinx.serialization.SerialName("output_index") val outputIndex: Int,
    @kotlinx.serialization.SerialName("call_id") val callId: String,
    val delta: String,
)

@kotlinx.serialization.Serializable
internal data class RealtimeResponseFunctionCallArgumentsDoneEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    @kotlinx.serialization.SerialName("response_id") val responseId: String,
    @kotlinx.serialization.SerialName("item_id") val itemId: String,
    @kotlinx.serialization.SerialName("output_index") val outputIndex: Int,
    @kotlinx.serialization.SerialName("call_id") val callId: String,
    val name: String,
    val arguments: String,
)

@kotlinx.serialization.Serializable
internal data class RealtimeResponseDoneEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    val response: RealtimeResponseResult,
)

@kotlinx.serialization.Serializable
internal data class RealtimeErrorEvent(
    @kotlinx.serialization.SerialName("event_id") val eventId: String? = null,
    val error: RealtimeError,
)
