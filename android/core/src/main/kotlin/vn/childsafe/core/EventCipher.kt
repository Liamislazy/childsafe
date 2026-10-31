package vn.childsafe.core

import java.nio.charset.StandardCharsets.UTF_8
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

data class Envelope(
    val epoch: String,
    val sequence: Long,
    val expiresAtMs: Long,
    val nonce: String,
    val ciphertext: String,
    val version: Int = 1,
) {
    init {
        require(version == 1)
        require(epoch.isNotEmpty() && epoch.length <= 80 && '\n' !in epoch && '/' !in epoch)
        require(sequence in 0..9_007_199_254_740_991L)
        require(expiresAtMs >= 0)
        require(EventCipher.decode(nonce).size == 12)
        require(EventCipher.decode(ciphertext).size in 16..16384)
    }

    fun aad(roomId: String, cameraId: String): ByteArray =
        "childsafe/1/$roomId/$cameraId/$epoch/$sequence/$expiresAtMs".toByteArray(UTF_8)
}

/** The QR supplies a 256-bit key. The relay never receives that key. */
object EventCipher {
    private val random = SecureRandom()
    fun newKey(): ByteArray = ByteArray(32).also(random::nextBytes)
    fun encode(bytes: ByteArray): String = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    fun decode(text: String): ByteArray = Base64.getUrlDecoder().decode(text)

    fun seal(key: ByteArray, plaintext: ByteArray, roomId: String, cameraId: String,
             epoch: String, sequence: Long, expiresAtMs: Long): Envelope {
        require(key.size == 32)
        require(plaintext.size <= 16368)
        val nonce = ByteArray(12).also(random::nextBytes)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(128, nonce))
        cipher.updateAAD("childsafe/1/$roomId/$cameraId/$epoch/$sequence/$expiresAtMs".toByteArray(UTF_8))
        return Envelope(epoch, sequence, expiresAtMs, encode(nonce), encode(cipher.doFinal(plaintext)))
    }

    fun open(key: ByteArray, message: Envelope, roomId: String, cameraId: String): ByteArray {
        require(key.size == 32)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "AES"),
            GCMParameterSpec(128, decode(message.nonce)))
        cipher.updateAAD(message.aad(roomId, cameraId))
        return cipher.doFinal(decode(message.ciphertext))
    }
}
