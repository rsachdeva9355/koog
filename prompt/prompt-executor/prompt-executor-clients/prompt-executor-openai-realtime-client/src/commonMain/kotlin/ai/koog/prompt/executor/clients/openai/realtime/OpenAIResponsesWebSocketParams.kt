package ai.koog.prompt.executor.clients.openai.realtime

import org.jetbrains.annotations.ApiStatus.Experimental

/**
 * Connection settings for [OpenAIResponsesWebSocketClient].
 *
 * @property baseUrl WebSocket host (no scheme, no path).
 * @property path WebSocket path for the Responses API endpoint.
 * @property connectTimeoutMillis Maximum time (ms) to wait for the WebSocket handshake.
 * @property socketTimeoutMillis Maximum idle time (ms) on the socket; 0 means no timeout.
 */
@Experimental
public data class OpenAIResponsesWebSocketSettings(
    val baseUrl: String = "api.openai.com",
    val path: String = "/v1/responses",
    val connectTimeoutMillis: Long = 30_000L,
    val socketTimeoutMillis: Long = 0L,
)
