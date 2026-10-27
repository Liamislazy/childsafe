package vn.childsafe.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.media.AudioAttributes
import android.media.RingtoneManager
import android.os.IBinder
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import org.json.JSONObject
import vn.childsafe.core.*

object MonitorData {
    val generation = java.util.concurrent.atomic.AtomicLong()
    @Volatile var cameras: Map<String, Pair<String, MonitorView>> = emptyMap()
    @Volatile var message = "Chưa bật máy theo dõi"
    @Volatile var running = false
}

open class MonitorService : Service() {
    private var worker: Thread? = null
    @Volatile private var stopped = false
    private val states = java.util.concurrent.ConcurrentHashMap<String, MonitorState>()
    private val lock = Any()
    private var generation = -1L
    private val main = Handler(Looper.getMainLooper())

    override fun onBind(intent: Intent?): IBinder? = null
    protected open fun monitorConfig(): DeviceConfig? = SecureConfig(this).load("monitor")

    private fun markOffline() {
        synchronized(lock) {
            val now = SystemClock.elapsedRealtime()
            MonitorData.cameras = MonitorData.cameras.mapValues { (id, old) ->
                old.first to (states[id]?.disconnected(now) ?: old.second.copy(
                    connection=Connection.OFFLINE, observation=Observation.UNOBSERVABLE, notify=false))
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") { stopSelf(); return START_NOT_STICKY }
        if (intent?.action == "MUTE") {
            synchronized(lock) { states.values.forEach { it.muteFor(SystemClock.elapsedRealtime(), 30_000) } }
            return START_NOT_STICKY
        }
        if (worker != null) return START_NOT_STICKY
        channels()
        startForeground(1, notification("Đang theo dõi", "Chờ dữ liệu mới từ máy camera", false))
        generation = MonitorData.generation.incrementAndGet()
        val current = generation
        markOffline()
        val config = try { monitorConfig() } catch (_: Exception) { null }
        if (config == null) {
            MonitorData.running=false
            MonitorData.message = "Cần ghép đôi lại máy theo dõi"
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(startId)
            return START_NOT_STICKY
        }
        stopped = false
        MonitorData.running = true
        worker = Thread({
            try { runMonitor(config) }
            finally {
                main.post {
                    if (current==generation && current==MonitorData.generation.get()) {
                        worker=null
                        MonitorData.running=false
                        markOffline()
                        stopForeground(STOP_FOREGROUND_REMOVE)
                        stopSelf(startId)
                    }
                }
            }
        }, "childsafe-monitor").also { it.start() }
        return START_NOT_STICKY // Never silently restart a disabled monitor.
    }

    protected open fun runMonitor(config: DeviceConfig) {
        try {
            val api = RelayApi(config.relayUrl, config.certificatePin)
            val key = EventCipher.decode(config.key)
            var cursor = 0L
            var bootstrap = true
            while (!stopped) {
                try {
                    val started = SystemClock.elapsedRealtime()
                    val path = "/v1/rooms/${config.roomId}/" + if (bootstrap) "snapshot" else "events?after=$cursor"
                    val response = api.call("GET", path, config.token)
                    if (stopped || generation != MonitorData.generation.get()) return
                    val now = SystemClock.elapsedRealtime()
                    val flightBudget = now - started // Conservative upper bound for response travel.
                    val serverNow = response.getLong("server_time_ms")
                    val cameras = response.getJSONArray("cameras")
                    val metadata = mutableMapOf<String, JSONObject>()
                    for (i in 0 until cameras.length()) {
                        val item = cameras.getJSONObject(i)
                        metadata[item.getString("id")] = item
                    }
                    synchronized(lock) {
                        states.keys.retainAll(metadata.keys)
                        metadata.forEach { (id, camera) ->
                            val state = states.getOrPut(id) { MonitorState() }
                            val epoch = camera.optString("epoch")
                            state.beginEpoch(epoch)
                            if (!camera.getBoolean("connected")) state.disconnected(now)
                        }
                        val messages = response.getJSONArray("messages")
                        for (i in 0 until messages.length()) {
                            val message = messages.getJSONObject(i)
                            val id = message.getString("camera_id")
                            val camera = metadata[id] ?: continue
                            val state = states[id] ?: continue
                            try {
                                val j = message.getJSONObject("envelope")
                                val e = Envelope(j.getString("epoch"), j.getLong("sequence"), j.getLong("expires_at_ms"),
                                    j.getString("nonce"), j.getString("ciphertext"), j.getInt("version"))
                                val age = serverNow - message.getLong("received_at_ms") + flightBudget
                                if (serverNow + flightBudget >= e.expiresAtMs || !camera.getBoolean("connected")) continue
                                val p = JSONObject(EventCipher.open(key, e, config.roomId, id).toString(Charsets.UTF_8))
                                if(p.getString("kind")=="test") {
                                    if(state.acceptTest(p.getString("event_id"),e.epoch,age,now,bootstrap)) {
                                        getSystemService(NotificationManager::class.java).notify(3,notification("ChildSafe · Thử đường cảnh báo",
                                            "Tín hiệu thử do người dùng gửi; không phải phát hiện nguy cơ.",true))
                                    }
                                    continue // Tests never change the real risk or observation state.
                                }
                                val observation = Observation.valueOf(p.getString("observation").uppercase())
                                val value = CameraResult(e.epoch, e.sequence, p.getString("event_id"),
                                    Risk.valueOf(p.getString("risk").uppercase()), observation, p.getString("reason"),
                                    p.getString("kind") == "alert")
                                val view = state.accept(value, camera.optString("epoch"), age, now, bootstrap)
                                if (view.notify) {
                                    getSystemService(NotificationManager::class.java).notify(id.hashCode(), notification(
                                        "${camera.getString("name")}: " + if (value.risk == Risk.DANGER) "Nguy cơ cao" else "Cần chú ý",
                                        "Kiểm tra trực tiếp. ${value.reason.take(160)}", true))
                                }
                            } catch (_: Exception) {
                                state.disconnected(now) // Authentication/format failure never becomes green.
                                MonitorData.message = "Không xác thực được dữ liệu của một máy camera"
                            }
                        }
                        MonitorData.cameras = metadata.mapValues { (id, item) -> item.getString("name") to states.getValue(id).view(now) }
                    }
                    cursor = response.getLong("cursor")
                    if (response.getBoolean("history_gap")) {
                        MonitorData.message = "Có khoảng dữ liệu bị gián đoạn; đang đồng bộ lại"
                        bootstrap = true
                    } else {
                        bootstrap = false
                        MonitorData.message = "Nhận sự kiện mã hóa; không truyền video"
                    }
                    getSystemService(NotificationManager::class.java).notify(1,notification("ChildSafe · Theo dõi",
                        if(MonitorData.cameras.values.any{it.second.connection==Connection.FRESH}) "Đang nhận cập nhật từ máy camera" else "Chưa có kết quả camera mới; cần kiểm tra kết nối và góc nhìn",false))
                    if (!response.getBoolean("more")) Thread.sleep(1000)
                } catch (e: InterruptedException) { return }
                catch (e: Exception) {
                    if (stopped || generation != MonitorData.generation.get()) return
                    markOffline()
                    MonitorData.message = if (e is RelayError && e.status == 401) "Quyền truy cập đã hết hoặc bị thu hồi; cần ghép đôi lại" else "Mất kết nối đến dịch vụ truyền cảnh báo"
                    getSystemService(NotificationManager::class.java).notify(1,notification("ChildSafe · Mất kết nối",MonitorData.message,false))
                    try { Thread.sleep(2000) } catch (_: InterruptedException) { return }
                }
            }
        } catch (_: Exception) {
            MonitorData.message = "Không mở được cấu hình theo dõi; cần ghép đôi lại"
        }
    }

    private fun channels() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("monitor", "Trạng thái theo dõi", NotificationManager.IMPORTANCE_LOW))
        val alerts = NotificationChannel("alerts", "Cảnh báo cần kiểm tra", NotificationManager.IMPORTANCE_HIGH)
        alerts.enableVibration(true)
        alerts.setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM),
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build())
        manager.createNotificationChannel(alerts)
    }

    private fun notification(title: String, text: String, alert: Boolean): Notification {
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return Notification.Builder(this, if (alert) "alerts" else "monitor")
            .setSmallIcon(android.R.drawable.ic_dialog_alert).setContentTitle(title).setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text)).setContentIntent(open)
            .setAutoCancel(alert).setOngoing(!alert).setVisibility(Notification.VISIBILITY_PRIVATE).build()
    }

    override fun onDestroy() {
        stopped = true
        worker?.interrupt()
        if (MonitorData.generation.compareAndSet(generation, generation + 1)) {
            MonitorData.running = false
            markOffline()
        }
        super.onDestroy()
    }
}
