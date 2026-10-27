package vn.childsafe.app

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.security.MessageDigest
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

/** No URL credentials, cleartext fallback, redirect or permissive TLS verifier. */
class RelayApi(baseUrl: String, certificatePin: String = "") {
    private val base = baseUrl.trimEnd('/')
    private val tls: SSLContext?
    init {
        val uri = URI(base)
        require(uri.scheme == "https" && !uri.host.isNullOrBlank()) { "Cần địa chỉ HTTPS" }
        require(uri.userInfo == null && uri.rawQuery == null && uri.rawFragment == null)
        tls = if (certificatePin.isEmpty()) null else {
            require(certificatePin.matches(Regex("[0-9a-fA-F]{64}"))) { "Fingerprint cần 64 ký tự hex" }
            val trustedPin = certificatePin.lowercase()
            val manager = object : X509TrustManager {
                override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
                override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {
                    throw CertificateException("Client certificates unsupported")
                }
                override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
                    if (chain.isEmpty()) throw CertificateException("No server certificate")
                    val actual = MessageDigest.getInstance("SHA-256").digest(chain[0].encoded)
                        .joinToString("") { "%02x".format(it.toInt() and 255) }
                    if (actual != trustedPin) throw CertificateException("Certificate fingerprint mismatch")
                    chain[0].checkValidity()
                }
            }
            SSLContext.getInstance("TLS").apply { init(null, arrayOf(manager), null) }
        }
    }

    fun call(method: String, path: String, token: String = "", body: JSONObject? = null): JSONObject {
        require(path.startsWith("/v1/") && !path.contains(".."))
        val c = URL(base + path).openConnection() as HttpsURLConnection
        c.instanceFollowRedirects = false
        c.requestMethod = method
        c.connectTimeout = 5000
        c.readTimeout = 5000
        tls?.let { c.sslSocketFactory = it.socketFactory }
        if (token.isNotEmpty()) c.setRequestProperty("Authorization", "Bearer $token")
        try {
            if (body != null) {
                val bytes = body.toString().toByteArray(Charsets.UTF_8)
                require(bytes.size <= 32768)
                c.doOutput = true
                c.setRequestProperty("Content-Type", "application/json")
                c.setFixedLengthStreamingMode(bytes.size)
                c.outputStream.use { it.write(bytes) }
            }
            val status = c.responseCode
            if (status !in 200..299) throw RelayError(status)
            if (status == HttpURLConnection.HTTP_NO_CONTENT) return JSONObject()
            val bytes = c.inputStream.use { stream ->
                val out = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(8192)
                while (true) {
                    val n = stream.read(buffer)
                    if (n < 0) break
                    if (out.size() + n > 4 * 1024 * 1024) error("Response exceeds limit")
                    out.write(buffer, 0, n)
                }
                out.toByteArray()
            }
            return JSONObject(bytes.toString(Charsets.UTF_8))
        } finally { c.disconnect() }
    }
}

class RelayError(val status: Int) : java.io.IOException("Relay HTTP $status")
