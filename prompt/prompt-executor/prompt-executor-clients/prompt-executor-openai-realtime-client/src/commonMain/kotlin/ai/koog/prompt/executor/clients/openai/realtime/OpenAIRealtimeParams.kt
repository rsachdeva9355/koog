package ai.koog.prompt.executor.clients.openai.realtime

import org.jetbrains.annotations.ApiStatus.Experimental

/**
 * Connection and behaviour settings for [OpenAIRealtimeClient].
 *
 * @property baseUrl WebSocket host for the Realtime API (no scheme, no path).
 * @property path WebSocket path for the Realtime endpoint.
 * @property connectTimeoutMillis Maximum time (ms) to wait for the WebSocket handshake.
 * @property socketTimeoutMillis Maximum idle time (ms) on the socket; 0 means no timeout.
 */
@Experimental
public data class OpenAIRealtimeClientSettings(
    val baseUrl: String = "api.openai.com",
    val path: String = "/v1/realtime",
    val connectTimeoutMillis: Long = 30_000L,
    val socketTimeoutMillis: Long = 0L,
)
