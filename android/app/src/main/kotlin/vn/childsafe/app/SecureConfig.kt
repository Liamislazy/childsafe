package vn.childsafe.app

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.AtomicFile
import org.json.JSONObject
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import vn.childsafe.core.EventCipher

data class DeviceConfig(val role: String, val relayUrl: String, val roomId: String,
                        val token: String, val key: String, val cameraId: String = "",
                        val certificatePin: String = "") {
    fun json() = JSONObject().put("role", role).put("relay_url", relayUrl).put("room_id", roomId)
        .put("token", token).put("key", key).put("camera_id", cameraId).put("certificate_pin", certificatePin)
    companion object {
        fun parse(j: JSONObject) = DeviceConfig(j.getString("role"), j.getString("relay_url"),
            j.getString("room_id"), j.getString("token"), j.getString("key"),
            j.optString("camera_id"), j.optString("certificate_pin"))
    }
}

/** Credentials are encrypted at rest with a non-exportable Android Keystore key. */
class SecureConfig(context: Context) {
    private val dir = context.filesDir
    private fun key(): SecretKey {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getKey("childsafe-config-v1", null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder("childsafe-config-v1", KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setKeySize(256).setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build())
        }.generateKey()
    }
    private fun file(role: String): AtomicFile {
        require(role in setOf("monitor", "camera"))
        return AtomicFile(java.io.File(dir, "$role.enc"))
    }
    @Synchronized fun save(config: DeviceConfig) {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val bytes = JSONObject().put("iv", EventCipher.encode(cipher.iv))
            .put("ciphertext", EventCipher.encode(cipher.doFinal(config.json().toString().toByteArray(Charsets.UTF_8))))
            .toString().toByteArray(Charsets.UTF_8)
        val f = file(config.role)
        val out = f.startWrite()
        try { out.write(bytes); f.finishWrite(out) } catch (e: Exception) { f.failWrite(out); throw e }
    }
    @Synchronized fun load(role: String): DeviceConfig? {
        val f = file(role)
        if (!f.baseFile.exists()) return null
        val wrapper = JSONObject(f.readFully().toString(Charsets.UTF_8))
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, EventCipher.decode(wrapper.getString("iv"))))
        return DeviceConfig.parse(JSONObject(cipher.doFinal(EventCipher.decode(wrapper.getString("ciphertext"))).toString(Charsets.UTF_8)))
    }
    @Synchronized fun clear(role: String) { file(role).delete() }
}
