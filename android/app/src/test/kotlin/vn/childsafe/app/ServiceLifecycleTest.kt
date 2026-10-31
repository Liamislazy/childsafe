package vn.childsafe.app

import android.Manifest
import android.app.NotificationManager
import android.content.Intent
import android.graphics.Bitmap
import android.os.Looper
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** Exercises the real Services with only model, camera hardware and network replaced.
 * These tests prove lifecycle behavior, not phone inference or background delivery. */
@RunWith(RobolectricTestRunner::class)
@Config(sdk=[28,35])
@LooperMode(LooperMode.Mode.PAUSED)
class ServiceLifecycleTest {
    private val controllers=mutableListOf<org.robolectric.android.controller.ServiceController<out android.app.Service>>()

    @Before fun before() {
        shadowOf(RuntimeEnvironment.getApplication()).grantPermissions(Manifest.permission.CAMERA)
        CameraData.running=false;CameraData.result="";CameraData.lastResultElapsed=0
        MonitorData.running=false;MonitorData.cameras=emptyMap();MonitorData.message=""
    }
    @After fun after() {
        controllers.asReversed().forEach{it.destroy()}
        idle()
    }
    private fun idle() { shadowOf(Looper.getMainLooper()).idle() }
    private fun await(message: String, condition: ()->Boolean) {
        val end=System.nanoTime()+TimeUnit.SECONDS.toNanos(5)
        do { idle(); if(condition()) return;Thread.sleep(5) } while(System.nanoTime()<end)
        fail(message)
    }
    private fun camera()=Robolectric.buildService(TestCamera::class.java).also{controllers.add(it)}.create()
    private fun monitor()=Robolectric.buildService(TestMonitor::class.java).also{controllers.add(it)}.create()
    private fun notifications()=shadowOf(RuntimeEnvironment.getApplication().getSystemService(NotificationManager::class.java))
    private fun previousDanger() {
        MonitorData.cameras=mapOf("old-camera" to ("Camera trước" to MonitorView(Connection.FRESH,Risk.DANGER,
            Observation.OBSERVABLE,"Bằng chứng trước",validUntilElapsedMs=Long.MAX_VALUE)))
    }

    @Test fun cameraModelFailureStopsWhileStillBoundAndClosesPublisher() {
        val c=camera();val s=c.get();s.failModel=true
        s.onBind(Intent()) // Keep the Activity binding alive while startup fails.
        c.startCommand(0,1)
        await("Failed model must stop runtime even while bound") {!CameraData.running && s.sink.closes.get()==1}
        assertEquals(0,s.activations.get())
        assertTrue(CameraData.message.contains("mô hình"))
        assertTrue(CameraData.result.isEmpty())
        assertTrue(shadowOf(s).isStoppedBySelf)
        assertNull(notifications().getNotification(2))
    }

    @Test fun cameraCanRetryOnTheSameBoundServiceAfterModelFailure() {
        val c=camera();val s=c.get();s.failModel=true;s.onBind(Intent())
        c.startCommand(0,1)
        await("Wait for failed startup cleanup") {!CameraData.running && s.sink.closes.get()==1}
        s.failModel=false
        c.startCommand(0,2)
        await("Second startup must activate camera") {CameraData.running && s.activations.get()==1}
        assertEquals("test-engine",CameraData.backend)
    }

    @Test fun cameraStopInvalidatesAnInFlightModelLoad() {
        val c=camera();val s=c.get();s.blockModel=true;s.onBind(Intent())
        c.startCommand(0,1)
        assertTrue(s.entered.await(5,TimeUnit.SECONDS))
        s.onStartCommand(Intent().setAction("STOP"),0,2)
        assertFalse(CameraData.running)
        s.release.countDown()
        await("Late-loaded model must be closed, not activated") {s.engine.closes.get()==1 && s.sink.closes.get()==1}
        assertEquals(0,s.activations.get())
        assertTrue(CameraData.backend.isEmpty())
        assertNull(notifications().getNotification(2))
    }

    @Test fun cameraOpenFailureStopsRuntimeAndReleasesModels() {
        val c=camera();val s=c.get();s.failCamera=true
        c.startCommand(0,1)
        await("Camera failure must stop and release models") {!CameraData.running && s.engine.closes.get()==1}
        assertTrue(CameraData.message.contains("camera"))
        assertNull(notifications().getNotification(2))
    }

    @Test fun duplicateCameraStartDoesNotLoadAnotherEngine() {
        val c=camera();val s=c.get()
        c.startCommand(0,1)
        await("First camera startup") {s.activations.get()==1}
        c.startCommand(0,2);idle()
        assertEquals(1,s.loads.get())
        assertEquals(0,s.engine.closes.get())
    }

    @Test fun monitorMissingConfigDropsForegroundAndKeepsLastRiskOffline() {
        previousDanger()
        val c=monitor();val s=c.get()
        c.startCommand(0,1)
        assertFalse(MonitorData.running)
        val old=MonitorData.cameras.getValue("old-camera").second
        assertEquals(Risk.DANGER,old.risk)
        assertEquals(Connection.OFFLINE,old.connection)
        assertFalse(old.green)
        assertNull(notifications().getNotification(1))
        assertTrue(shadowOf(s).isStoppedBySelf)
    }

    @Test fun monitorCanRetryAfterWorkerReturns() {
        val c=monitor();val s=c.get();s.config=config("monitor")
        c.startCommand(0,1)
        await("First worker cleanup") {s.calls.get()==1 && !MonitorData.running}
        c.startCommand(0,2)
        await("Second worker must be started") {s.calls.get()==2 && !MonitorData.running}
        assertNull(notifications().getNotification(1))
    }

    @Test fun monitorStopCannotRepaintOldFreshDataOrLeaveRunningFlag() {
        previousDanger()
        val c=monitor();val s=c.get();s.config=config("monitor");s.block=true
        c.startCommand(0,1)
        assertTrue(s.entered.await(5,TimeUnit.SECONDS))
        c.destroy();controllers.remove(c)
        await("Interrupted monitor must exit") {s.exited.count==0L}
        val old=MonitorData.cameras.getValue("old-camera").second
        assertFalse(MonitorData.running)
        assertEquals(Connection.OFFLINE,old.connection)
        assertEquals(Risk.DANGER,old.risk)
    }

    @Test fun invalidRelayConfigurationUsesRealWorkerAndCanRetry() {
        val c=Robolectric.buildService(InvalidConfigMonitor::class.java).also{controllers.add(it)}.create()
        c.startCommand(0,1)
        await("Invalid HTTPS configuration must finish worker") {!MonitorData.running}
        assertTrue(MonitorData.message.contains("ghép đôi"))
        assertNull(notifications().getNotification(1))
        // The same Service used to retain a dead worker Thread and ignore this start.
        val first=MonitorData.generation.get()
        c.startCommand(0,2)
        await("Retry must create another generation") {MonitorData.generation.get()>first && !MonitorData.running}
    }

    class TestEngine: CameraEngine {
        val closes=AtomicInteger()
        override val backend="test-engine"
        override fun process(bitmap: Bitmap,captured: Long): VisionOutput = error("No camera frames in lifecycle test")
        override fun reset() {}
        override fun close() {closes.incrementAndGet()}
    }
    class TestSink: CameraEventSink {
        val closes=AtomicInteger()
        override val error=""
        override fun submit(state: JSONObject,captured: Long,alert: Boolean) {}
        override fun close() {closes.incrementAndGet()}
    }
    class TestCamera: CameraNodeService() {
        val engine=TestEngine();val sink=TestSink()
        val entered=CountDownLatch(1);val release=CountDownLatch(1)
        val loads=AtomicInteger();val activations=AtomicInteger()
        @Volatile var failModel=false
        @Volatile var blockModel=false
        var failCamera=false
        override fun cameraConfig()=config("camera")
        override fun createPublisher(config: DeviceConfig): CameraEventSink=sink
        override fun createEngine(): CameraEngine {
            loads.incrementAndGet();entered.countDown()
            if(blockModel) check(release.await(5,TimeUnit.SECONDS))
            if(failModel) error("Synthetic model startup failure")
            return engine
        }
        override fun activateCamera() {
            activations.incrementAndGet()
            if(failCamera) error("Synthetic unavailable camera")
        }
    }
    class TestMonitor: MonitorService() {
        var config: DeviceConfig?=null
        val calls=AtomicInteger();val entered=CountDownLatch(1);val exited=CountDownLatch(1)
        var block=false
        override fun monitorConfig()=config
        override fun runMonitor(config: DeviceConfig) {
            calls.incrementAndGet();entered.countDown()
            try {if(block) CountDownLatch(1).await()} catch(_:InterruptedException) {}
            finally {exited.countDown()}
        }
    }
    class InvalidConfigMonitor: MonitorService() {
        override fun monitorConfig()=config("monitor").copy(relayUrl="http://invalid.example")
    }
    companion object {
        fun config(role: String)=DeviceConfig(role,"https://relay.invalid","test-room","test-token",
            EventCipher.encode(ByteArray(32)),"test-camera")
    }
}
