package vn.childsafe.app

import android.app.Notification
import android.app.NotificationManager
import android.os.Looper
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode
import vn.childsafe.core.*
import java.time.Duration
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** Runs the actual receive/decrypt/filter/Android-notify path. Only HTTP is replaced;
 * notification sounds are inspected as configuration, not played on real hardware. */
@RunWith(RobolectricTestRunner::class)
@Config(sdk=[28,35])
@LooperMode(LooperMode.Mode.PAUSED)
class MonitorDeliveryTest {
    private var controller: org.robolectric.android.controller.ServiceController<ReceiveMonitor>?=null
    @Before fun before() { MonitorData.running=false;MonitorData.cameras=emptyMap() }
    @After fun after() {controller?.destroy();shadowOf(Looper.getMainLooper()).idle()}
    private fun manager()=RuntimeEnvironment.getApplication().getSystemService(NotificationManager::class.java)
    private fun notification(id: Int)=shadowOf(manager()).getNotification(id)
    private fun start(vararg responses: JSONObject) {
        val c=Robolectric.buildService(ReceiveMonitor::class.java).create()
        controller=c
        c.get().api.responses=responses.toList()
        c.startCommand(0,1)
        assertTrue("Receiver must consume the test responses",c.get().api.tail.await(5,TimeUnit.SECONDS))
        shadowOf(Looper.getMainLooper()).idle()
    }
    private fun view()=MonitorData.cameras.getValue(cameraId).second

    @Test fun newEncryptedDangerCreatesHighImportanceAndroidNotification() {
        start(response(message(0,"normal","status")),response(message(1,"danger","alert")))
        val alert=notification(cameraId.hashCode())
        assertNotNull(alert)
        assertEquals("alerts",alert.channelId)
        assertTrue(alert.extras.getString(Notification.EXTRA_TITLE)!!.contains("Nguy cơ cao"))
        val channel=manager().getNotificationChannel("alerts")
        assertEquals(NotificationManager.IMPORTANCE_HIGH,channel.importance)
        assertNotNull(channel.sound)
        assertEquals(Risk.DANGER,view().risk)
    }

    @Test fun snapshotOfAnOldDangerDoesNotReplayAnAlarm() {
        start(response(message(0,"danger","alert")))
        assertEquals(Risk.DANGER,view().risk)
        assertNull(notification(cameraId.hashCode()))
    }

    @Test fun testSignalCreatesItsOwnNotificationWithoutChangingActualRisk() {
        start(response(message(0,"normal","status")),response(message(1,"normal","test")))
        val test=notification(3)
        assertNotNull(test)
        assertTrue(test.extras.getString(Notification.EXTRA_TEXT)!!.contains("không phải phát hiện nguy cơ"))
        assertEquals(Risk.NORMAL,view().risk)
        assertNull(notification(cameraId.hashCode()))
    }

    @Test fun expiredDangerCannotCreateAnAndroidAlert() {
        start(response(message(0,"normal","status")),response(message(1,"danger","alert",expired=true)))
        assertEquals(Risk.NORMAL,view().risk)
        assertNull(notification(cameraId.hashCode()))
    }

    @Test fun tamperedDangerCannotCreateAnAlertAndReportsAuthenticationFailure() {
        val invalid=message(1,"danger","alert")
        val envelope=invalid.getJSONObject("envelope")
        val bytes=EventCipher.decode(envelope.getString("ciphertext"))
        bytes[0]=(bytes[0].toInt() xor 1).toByte()
        envelope.put("ciphertext",EventCipher.encode(bytes))
        start(response(message(0,"normal","status")),response(invalid))
        assertNull(notification(cameraId.hashCode()))
        assertEquals(Risk.NORMAL,view().risk)
        assertEquals(Connection.OFFLINE,view().connection)
        assertTrue(MonitorData.message.contains("Không xác thực"))
    }

    @Test fun foregroundStatusExpiresWhileHttpIsBlockedAndPreservesDanger() {
        start(response(message(0,"danger","status")))
        assertEquals(Connection.FRESH,view().connection)
        shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(3000))
        assertEquals(Connection.STALE,view().connection)
        assertEquals(Risk.DANGER,view().risk)
        assertFalse(view().green)
        assertTrue(notification(1).extras.getString(Notification.EXTRA_TEXT)!!.contains("Chưa có kết quả camera mới"))
        assertNull(notification(cameraId.hashCode()))
    }

    class ReceiveMonitor: MonitorService() {
        val api=Responses()
        override fun monitorConfig()=DeviceConfig("monitor","https://relay.invalid","test-room","test-token",EventCipher.encode(key))
        override fun relayApi(config: DeviceConfig): RelayApi=api
    }
    class Responses: RelayApi("https://relay.invalid") {
        var responses=listOf<JSONObject>()
        val calls=AtomicInteger();val tail=CountDownLatch(1)
        override fun call(method: String,path: String,token: String,body: JSONObject?): JSONObject {
            val i=calls.getAndIncrement()
            if(i<responses.size) return responses[i]
            tail.countDown()
            CountDownLatch(1).await() // Simulate a request pending beyond the freshness deadline.
            error("Interrupted by service shutdown")
        }
    }
    companion object {
        private val key=ByteArray(32) { it.toByte() } // Public fixture only.
        private const val cameraId="fixture-camera"
        private const val epoch="fixture-epoch"
        private const val serverNow=1_000_000L
        private fun message(sequence: Long,risk: String,kind: String,expired: Boolean=false): JSONObject {
            val data=JSONObject().put("event_id","event-$sequence").put("kind",kind).put("risk",risk)
                .put("observation","degraded").put("reason","Fixture; no real camera inference")
                .put("captured_at_ms",SystemClock.elapsedRealtime()).put("people",JSONArray())
            val e=EventCipher.seal(key,data.toString().toByteArray(),"test-room",cameraId,epoch,sequence,
                serverNow+if(expired) -1 else 2500)
            return JSONObject().put("camera_id",cameraId).put("received_at_ms",serverNow)
                .put("envelope",JSONObject().put("version",1).put("epoch",e.epoch).put("sequence",e.sequence)
                    .put("expires_at_ms",e.expiresAtMs).put("nonce",e.nonce).put("ciphertext",e.ciphertext))
        }
        private fun response(message: JSONObject)=JSONObject().put("server_time_ms",serverNow)
            .put("cameras",JSONArray().put(JSONObject().put("id",cameraId).put("name","Camera thử")
                .put("epoch",epoch).put("connected",true)))
            .put("messages",JSONArray().put(message)).put("cursor",1).put("history_gap",false).put("more",true)
    }
}
