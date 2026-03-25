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
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.parseToJsonElement
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import org.jetbrains.annotations.ApiStatus.Experimental
import kotlin.time.Clock

private val logger = KotlinLogging.logger {}

/**
 * LLM client that connects to the
 * [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) via WebSocket,
 * supporting text responses, function/tool calls, and **reasoning** (chain-of-thought) output.
 *
 * Unlike [OpenAIRealtimeClient] (which targets the legacy Realtime API at `/v1/realtime`),
 * this client uses the **WebSocket mode** of the Responses API at `wss://api.openai.com/v1/responses`.
 * This endpoint supports the full range of Responses API models, including reasoning-capable ones
 * such as **o3** and **o4-mini**.
 *
 * Each call to [execute] or [executeStreaming] opens a dedicated WebSocket connection, sends a
 * single `response.create` event with the complete conversation context, streams the resulting
 * [StreamFrame]s back to the caller, and closes the connection when the response is complete.
 *
 * This client implements [LLMClient] and is fully compatible with the koog agent framework.
 *
 * ### Supported features
 * - Text responses streamed as [StreamFrame.TextDelta] / [StreamFrame.TextComplete]
 * - Function/tool calls streamed as [StreamFrame.ToolCallDelta] / [StreamFrame.ToolCallComplete]
 * - Reasoning streamed as [StreamFrame.ReasoningDelta] / [StreamFrame.ReasoningComplete]
 * - `reasoning.encrypted_content` included automatically for stateless multi-turn reasoning
 * - System instructions via [Message.System] (sent as `developer` role)
 * - Reasoning history via [Message.Reasoning] (sent with encrypted tokens when available)
 *
 * ### Not supported
 * - Audio input/output
 * - Content moderation
 *
 * ### Usage
 * ```kotlin
 * val client = OpenAIResponsesWebSocketClient(apiKey = System.getenv("OPENAI_API_KEY"))
 *
 * val executor = SimplePromptExecutorStrategy(client)
 * val agent = AIAgent(promptExecutor = executor, ...) { ... }
 * ```
 *
 * @param apiKey OpenAI API key.
 * @param settings Connection configuration. See [OpenAIResponsesWebSocketSettings].
 * @param baseClient Ktor [HttpClient] with [WebSockets] installed.
 * @param clock Clock used to populate [ResponseMetaInfo] timestamps.
 * @param toolsConverter Schema generator for tool parameter schemas.
 */
@Experimental
public class OpenAIResponsesWebSocketClient @JvmOverloads constructor(
    private val apiKey: String,
    private val settings: OpenAIResponsesWebSocketSettings = OpenAIResponsesWebSocketSettings(),
    baseClient: HttpClient = HttpClient { install(WebSockets) },
    private val clock: Clock = Clock.System,
    private val toolsConverter: OpenAICompatibleToolDescriptorSchemaGenerator =
        OpenAICompatibleToolDescriptorSchemaGenerator(),
) : LLMClient() {

    override val clientName: String get() = "OpenAIResponsesWebSocketClient"

    private val httpClient: HttpClient = baseClient

    private val json: Json = Json {
        classDiscriminator = "type"
        ignoreUnknownKeys = true
        encodeDefaults = false
        explicitNulls = false
    }

    // -------------------------------------------------------------------------
    // LLMClient implementation
    // -------------------------------------------------------------------------

    override fun llmProvider(): LLMProvider = LLMProvider.OpenAI

    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): List<Message.Response> =
        executeStreaming(prompt, model, tools).toList().toMessageResponses()

    /**
     * Opens a WebSocket connection to the OpenAI Responses API, sends a `response.create` event
     * with the full conversation context, and returns a cold [Flow] of [StreamFrame]s.
     *
     * Reasoning encrypted content is requested automatically via the `include` parameter, enabling
     * stateless multi-turn conversations with reasoning models.
     */
    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Flow<StreamFrame> = channelFlow {
        logger.debug { "Opening Responses API WebSocket session for model=${model.id}" }

        try {
            httpClient.wss(
                host = settings.baseUrl,
                path = settings.path,
                request = {
                    headers.append(HttpHeaders.Authorization, "Bearer $apiKey")
                },
            ) {
                // Build and send the response.create event with the full request.
                val createEvent = buildResponseCreateEvent(prompt, model, tools)
                val eventJson = json.encodeToString(createEvent)
                logger.debug { "→ ResponsesWS: $eventJson" }
                send(Frame.Text(eventJson))

                // Accumulated reasoning text/summary per outputIndex for ReasoningComplete.
                val reasoningText = mutableMapOf<Int, MutableList<String>>()
                val reasoningSummary = mutableMapOf<Int, MutableList<String>>()
                // call_id → function name, populated from response.output_item.added.
                val functionCallNames = mutableMapOf<String, String>()

                try {
                    for (frame in incoming) {
                        if (frame !is Frame.Text) continue

                        val rawText = frame.readText()
                        logger.debug { "← ResponsesWS: $rawText" }
                        val event = decodeServerEvent(rawText) ?: continue

                        when (event) {
                            is RealtimeErrorEvent ->
                                throw LLMClientException(
                                    clientName = clientName,
                                    message = buildString {
                                        append("OpenAI Responses WebSocket error")
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

                            is RealtimeResponseOutputTextDeltaEvent ->
                                send(StreamFrame.TextDelta(event.delta, event.outputIndex))

                            is RealtimeResponseOutputTextDoneEvent ->
                                send(StreamFrame.TextComplete(event.text, event.outputIndex))

                            is RealtimeResponseReasoningTextDeltaEvent -> {
                                reasoningText.getOrPut(event.outputIndex) { mutableListOf() }
                                    .add(event.delta)
                                send(StreamFrame.ReasoningDelta(text = event.delta, index = event.outputIndex))
                            }

                            is RealtimeResponseReasoningSummaryTextDeltaEvent -> {
                                reasoningSummary.getOrPut(event.outputIndex) { mutableListOf() }
                                    .add(event.delta)
                                send(StreamFrame.ReasoningDelta(summary = event.delta, index = event.outputIndex))
                            }

                            is RealtimeResponseOutputItemDoneEvent -> {
                                val item = event.item
                                if (item.type == "reasoning") {
                                    val idx = event.outputIndex
                                    val summaryTexts = item.summary?.map { it.text }
                                        ?: reasoningSummary[idx]
                                    send(
                                        StreamFrame.ReasoningComplete(
                                            text = reasoningText[idx] ?: emptyList(),
                                            summary = summaryTexts,
                                            encrypted = item.encryptedContent,
                                            index = idx,
                                        )
                                    )
                                }
                            }

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
                                break
                            }

                            else -> {}
                        }
                    }
                } catch (e: ClosedReceiveChannelException) {
                    logger.debug { "Responses WebSocket incoming channel closed unexpectedly" }
                }
            }
        } catch (e: LLMClientException) {
            throw e
        } catch (e: Exception) {
            throw LLMClientException(
                clientName = clientName,
                message = "OpenAI Responses WebSocket error: ${e.message}",
                cause = e,
            )
        }
    }.requireEndFrame()

    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        throw UnsupportedOperationException(
            "Content moderation is not supported by the OpenAI Responses WebSocket API. " +
                "Use OpenAILLMClient for moderation."
        )

    override fun close() {
        logger.debug { "Closing $clientName" }
        httpClient.close()
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Builds the `response.create` WebSocket client event with the full Responses API request.
     * Includes `reasoning.encrypted_content` in the `include` list so that reasoning items
     * carry their encrypted token representation — required for stateless multi-turn reasoning.
     */
    private fun buildResponseCreateEvent(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): ResponsesWsCreateEvent {
        val instructions = prompt.messages
            .filterIsInstance<Message.System>()
            .lastOrNull()?.content

        val input = buildJsonArray {
            val pendingCalls = mutableListOf<JsonObject>()

            fun flushPendingCalls() {
                pendingCalls.forEach { add(it) }
                pendingCalls.clear()
            }

            for (message in prompt.messages) {
                when (message) {
                    is Message.System -> {
                        // Sent as `instructions` at the top-level; skip as conversation item.
                    }

                    is Message.User -> {
                        flushPendingCalls()
                        add(buildJsonObject {
                            put("type", "message")
                            put("role", "user")
                            putJsonArray("content") {
                                add(buildJsonObject {
                                    put("type", "input_text")
                                    put("text", message.content)
                                })
                            }
                        })
                    }

                    is Message.Assistant -> {
                        flushPendingCalls()
                        add(buildJsonObject {
                            put("type", "message")
                            put("role", "assistant")
                            putJsonArray("content") {
                                add(buildJsonObject {
                                    put("type", "output_text")
                                    put("text", message.content)
                                })
                            }
                        })
                    }

                    is Message.Reasoning -> {
                        flushPendingCalls()
                        add(buildJsonObject {
                            put("type", "reasoning")
                            if (message.id != null) put("id", message.id)
                            if (message.encrypted != null) put("encrypted_content", message.encrypted)
                            // Include summary so the model can display its chain-of-thought.
                            val summaryParts = message.summary
                            if (!summaryParts.isNullOrEmpty()) {
                                putJsonArray("summary") {
                                    summaryParts.forEach { part ->
                                        add(buildJsonObject {
                                            put("type", "summary_text")
                                            put("text", part.text)
                                        })
                                    }
                                }
                            }
                        })
                    }

                    is Message.Tool.Call -> {
                        pendingCalls += buildJsonObject {
                            put("type", "function_call")
                            put("name", message.tool)
                            put("call_id", message.id ?: message.tool)
                            put("arguments", message.content)
                        }
                    }

                    is Message.Tool.Result -> {
                        flushPendingCalls()
                        add(buildJsonObject {
                            put("type", "function_call_output")
                            put("call_id", message.id ?: message.tool)
                            put("output", message.content)
                        })
                    }
                }
            }
            flushPendingCalls()
        }

        val toolsJson = if (tools.isEmpty()) null else tools.map { descriptor ->
            buildJsonObject {
                put("type", "function")
                put("name", descriptor.name)
                put("description", descriptor.description)
                put("parameters", toolsConverter.generate(descriptor))
            }
        }

        return ResponsesWsCreateEvent(
            response = ResponsesWsRequest(
                model = model.id,
                input = input,
                instructions = instructions,
                tools = toolsJson,
                toolChoice = if (tools.isEmpty()) null else JsonPrimitive("auto"),
                temperature = prompt.params.temperature,
                include = listOf("reasoning.encrypted_content"),
            )
        )
    }

    private fun decodeServerEvent(text: String): Any? = try {
        val element = json.parseToJsonElement(text)
        val type = element.jsonObject["type"]?.jsonPrimitive?.content ?: return null
        when (type) {
            "response.output_item.added" ->
                json.decodeFromJsonElement<RealtimeResponseOutputItemAddedEvent>(element)
            "response.output_item.done" ->
                json.decodeFromJsonElement<RealtimeResponseOutputItemDoneEvent>(element)
            "response.output_text.delta" ->
                json.decodeFromJsonElement<RealtimeResponseOutputTextDeltaEvent>(element)
            "response.output_text.done" ->
                json.decodeFromJsonElement<RealtimeResponseOutputTextDoneEvent>(element)
            "response.reasoning_text.delta" ->
                json.decodeFromJsonElement<RealtimeResponseReasoningTextDeltaEvent>(element)
            "response.reasoning_summary_text.delta" ->
                json.decodeFromJsonElement<RealtimeResponseReasoningSummaryTextDeltaEvent>(element)
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
        logger.debug(e) { "Failed to decode Responses WebSocket server event: $text" }
        null
    }
}

// -------------------------------------------------------------------------
// Client event models
// -------------------------------------------------------------------------

/** Sealed base for all client events sent over the WebSocket. */
@Serializable
private sealed class ResponsesWsClientEvent

/**
 * The single client event used to initiate a response.
 * Carries the complete Responses API request as [response].
 */
@Serializable
@SerialName("response.create")
private data class ResponsesWsCreateEvent(
    val response: ResponsesWsRequest,
) : ResponsesWsClientEvent()

/**
 * The Responses API request body, embedded inside [ResponsesWsCreateEvent].
 *
 * @property model ID of the model to use (e.g. `"o3-mini"`, `"gpt-4o"`).
 * @property input Conversation history as a JSON array of input items.
 * @property instructions Optional system-level instructions (sent as `developer` role).
 * @property tools Optional function tool definitions.
 * @property toolChoice Tool selection strategy (`"auto"`, `"none"`, `"required"`, or a specific tool).
 * @property temperature Sampling temperature.
 * @property include List of extra fields to include; use `"reasoning.encrypted_content"` for
 *   stateless multi-turn reasoning.
 */
@Serializable
private data class ResponsesWsRequest(
    val model: String,
    val input: JsonArray,
    val instructions: String? = null,
    val tools: List<JsonObject>? = null,
    @SerialName("tool_choice") val toolChoice: JsonElement? = null,
    val temperature: Double? = null,
    val include: List<String>? = null,
)
