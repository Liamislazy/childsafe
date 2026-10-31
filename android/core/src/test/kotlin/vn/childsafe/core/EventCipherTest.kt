package vn.childsafe.core

import org.junit.Assert.*
import org.junit.Test
import javax.crypto.AEADBadTagException

class EventCipherTest {
    @Test fun decryptsPythonInteroperabilityFixture() {
        val p = java.util.Properties()
        javaClass.classLoader!!.getResourceAsStream("companion-v1.properties")!!.use(p::load)
        val message = Envelope(p.getProperty("epoch"), p.getProperty("sequence").toLong(),
            p.getProperty("expires_at_ms").toLong(), p.getProperty("nonce"), p.getProperty("ciphertext"))
        val actual = EventCipher.open(EventCipher.decode(p.getProperty("test_key")), message,
            p.getProperty("room_id"), p.getProperty("camera_id"))
        assertArrayEquals(EventCipher.decode(p.getProperty("plaintext")), actual)
    }
    @Test fun roundTrip() {
        val key = EventCipher.newKey()
        val plain = "Test event only, no image".toByteArray()
        val m = EventCipher.seal(key, plain, "room", "camera", "epoch", 0, 1000)
        assertArrayEquals(plain, EventCipher.open(key, m, "room", "camera"))
        assertNotEquals(m.nonce, EventCipher.seal(key, plain, "room", "camera", "epoch", 0, 1000).nonce)
    }

    @Test(expected = AEADBadTagException::class) fun swappedCameraRejected() {
        val key = EventCipher.newKey()
        val m = EventCipher.seal(key, "event".toByteArray(), "room", "camera", "epoch", 0, 1000)
        EventCipher.open(key, m, "room", "another-camera")
    }

    @Test(expected = AEADBadTagException::class) fun changedExpiryRejected() {
        val key = EventCipher.newKey()
        val m = EventCipher.seal(key, "event".toByteArray(), "room", "camera", "epoch", 0, 1000)
        EventCipher.open(key, m.copy(expiresAtMs = 1001), "room", "camera")
    }
}
