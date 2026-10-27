package vn.childsafe.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.google.zxing.BarcodeFormat
import com.google.zxing.MultiFormatWriter
import org.json.JSONObject
import vn.childsafe.core.*
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val handler = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private lateinit var body: LinearLayout
    private lateinit var message: TextView
    private var tab = "Theo dõi"
    private var resumed = false
    private var scanKind="pair"
    private val bg = Color.rgb(246,247,249)
    private val ink = Color.rgb(29,35,48)
    private val muted = Color.rgb(102,112,133)
    private val blue = Color.rgb(59,111,216)
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startMonitor() else message.text = "Chưa được cấp quyền thông báo; chưa bật theo dõi nền."
    }
    private val scanPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) scan.launch(Intent(this, PairScanActivity::class.java).putExtra("kind",scanKind))
        else message.text = "Cần quyền camera để quét mã; bạn vẫn có thể dán mã ghép đôi."
    }
    private val scan = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) result.data?.getStringExtra("pairing")?.let { code ->
            if(code.startsWith("childsafe-relay:")) {
                try {
                    val data=JSONObject(EventCipher.decode(code.removePrefix("childsafe-relay:")).toString(Charsets.UTF_8))
                    require(data.getInt("version")==1)
                    createMonitor(data.getString("relay_url"),data.getString("certificate_pin"))
                } catch(_:Exception){message.text="Mã cấu hình máy chủ không hợp lệ."}
            } else pairCamera(code)
        }
    }
    private val refresh = object : Runnable {
        override fun run() { if (resumed && tab == "Theo dõi") renderMonitor(); if (resumed) handler.postDelayed(this, 1000) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(bg) }
        root.setOnApplyWindowInsetsListener { view, insets ->
            if (Build.VERSION.SDK_INT >= 30) {
                val bars = insets.getInsets(WindowInsets.Type.systemBars())
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            } else { @Suppress("DEPRECATION") view.setPadding(0, insets.systemWindowInsetTop, 0, insets.systemWindowInsetBottom) }
            insets
        }
        root.addView(text("ChildSafe", 22, true).apply { setPadding(dp(16),dp(16),dp(16),0) })
        root.addView(text("Hỗ trợ người chăm sóc", 16).apply { setPadding(dp(16),0,dp(16),dp(8)) })
        root.addView(text("Nguyên mẫu đang kiểm chứng", 12).apply { setPadding(dp(16),0,dp(16),dp(8)); setTextColor(muted) })
        val scroll = ScrollView(this)
        body = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(16),dp(8),dp(16),dp(16)) }
        scroll.addView(body)
        root.addView(scroll, LinearLayout.LayoutParams(-1,0,1f))
        message = text("",14).apply { setPadding(dp(16),dp(8),dp(16),dp(8)) }
        root.addView(message)
        val nav = LinearLayout(this).apply { setBackgroundColor(Color.WHITE) }
        listOf("Theo dõi", "Ghép đôi", "Cài đặt").forEach { label ->
            nav.addView(Button(this).apply { this.text = label; minHeight = dp(56); isAllCaps = false
                setOnClickListener { tab = label; render() } }, LinearLayout.LayoutParams(0,-2,1f))
        }
        root.addView(nav)
        setContentView(root)
        render()
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun text(value: String, size: Int = 16, bold: Boolean = false) = TextView(this).apply {
        this.text = value; textSize = size.toFloat(); setTextColor(ink)
        if (bold) setTypeface(typeface, Typeface.BOLD)
    }
    private fun card(title: String, description: String, color: Int = ink) {
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(16),dp(16),dp(16),dp(16))
            background = GradientDrawable().apply { setColor(Color.WHITE); cornerRadius = dp(12).toFloat(); setStroke(dp(1), Color.rgb(227,230,235)) } }
        box.addView(text(title,18,true).apply { setTextColor(color) })
        box.addView(text(description,16).apply { setPadding(0,dp(8),0,0) })
        body.addView(box, LinearLayout.LayoutParams(-1,-2).apply { bottomMargin = dp(12) })
    }
    private fun button(label: String, action: () -> Unit) {
        body.addView(Button(this).apply { text = label; isAllCaps = false; minHeight = dp(48); setTextColor(blue); setOnClickListener { action() } })
    }
    private fun input(hintText: String): EditText = EditText(this).apply {
        hint = hintText; setSingleLine(true); textSize = 16f
        body.addView(this, LinearLayout.LayoutParams(-1,-2).apply { bottomMargin = dp(8) })
    }
    private fun render() { when(tab) { "Theo dõi" -> renderMonitor(); "Ghép đôi" -> renderPairing(); else -> renderSettings() } }

    private fun renderMonitor() {
        body.removeAllViews()
        card("Trạng thái hệ thống", MonitorData.message)
        if (MonitorData.cameras.isEmpty()) card("Chưa có dữ liệu camera", "Ghép đôi máy camera, sau đó bật theo dõi. Không có dữ liệu không đồng nghĩa với an toàn.")
        MonitorData.cameras.forEach { (_, item) ->
            val (name, stored) = item
            val v = stored.atTime(android.os.SystemClock.elapsedRealtime())
            val riskLabel = when(v.risk) { Risk.DANGER -> "Nguy cơ cao"; Risk.CAUTION -> "Cần chú ý"; else -> if(v.green) "Chưa ghi nhận nguy cơ" else "Chưa quan sát rõ" }
            val label = if(v.connection != Connection.FRESH) {
                val lost = if(v.connection == Connection.OFFLINE) "Mất kết nối" else "Dữ liệu đã cũ"
                if(v.risk != null && v.risk != Risk.NORMAL) "$riskLabel gần nhất · $lost" else lost
            } else riskLabel
            val color = when(v.risk) {
                Risk.DANGER -> Color.rgb(211,58,44); Risk.CAUTION -> Color.rgb(217,138,0)
                else -> if(v.green) Color.rgb(31,157,85) else muted
            }
            val observation = when(v.observation) { Observation.OBSERVABLE -> "Quan sát đủ rõ theo ngưỡng hiện tại"; Observation.DEGRADED -> "Quan sát hạn chế"; Observation.UNOBSERVABLE -> "Chưa quan sát được" }
            card("$name · $label", "$observation\n${v.reason}" + if(v.risk != null && v.risk != Risk.NORMAL) "\nKiểm tra trực tiếp ngay khi cần. Trạng thái nguy cơ gần nhất vẫn được giữ khi mất kết nối." else "", color)
        }
        button(if(MonitorData.running) "Tắt theo dõi" else "Bật theo dõi") {
            if(MonitorData.running) { stopService(Intent(this, MonitorService::class.java)); renderMonitor() }
            else requestMonitor()
        }
        if(MonitorData.running) button("Tạm ngừng âm báo mới 30 giây") { startService(Intent(this, MonitorService::class.java).setAction("MUTE")); message.text = "Chỉ tạm ngừng âm báo; mức nguy cơ vẫn được giữ." }
    }

    private fun renderPairing() {
        body.removeAllViews()
        card("Chọn vai trò thiết bị", "Máy theo dõi nhận sự kiện. Máy camera phân tích hình ảnh. Mã ghép đôi chỉ chia sẻ với thiết bị của bạn.")
        val url = input("Địa chỉ dịch vụ HTTPS")
        val pin = input("Fingerprint chứng chỉ LAN (nếu có)")
        button("Quét mã cấu hình máy chủ trên máy tính") {
            if(CameraData.running){message.text="Dừng máy camera trước khi quét mã.";return@button}
            scanKind="relay"
            if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)
                scan.launch(Intent(this,PairScanActivity::class.java).putExtra("kind",scanKind))
            else scanPermission.launch(Manifest.permission.CAMERA)
        }
        button("Tạo máy theo dõi") {
            val address = url.text.toString().trim()
            val fingerprint = pin.text.toString().trim()
            createMonitor(address,fingerprint)
        }
        button("Tạo mã QR ghép đôi camera (2 phút)") { network {
            val c = SecureConfig(this).load("monitor") ?: error("Chưa tạo máy theo dõi")
            val invitation = RelayApi(c.relayUrl,c.certificatePin).call("POST", "/v1/rooms/${c.roomId}/invitations",c.token)
            val payload = JSONObject().put("version",1).put("relay_url",c.relayUrl).put("room_id",c.roomId)
                .put("invitation",invitation.getString("invitation")).put("key",c.key)
                .put("certificate_pin",c.certificatePin).put("expires_at_ms",invitation.getLong("expires_at_ms"))
            val code = "childsafe-pair:"+EventCipher.encode(payload.toString().toByteArray(Charsets.UTF_8))
            val bits = MultiFormatWriter().encode(code,BarcodeFormat.QR_CODE,600,600)
            val bitmap = Bitmap.createBitmap(600,600,Bitmap.Config.ARGB_8888)
            for(y in 0 until 600) for(x in 0 until 600) bitmap.setPixel(x,y,if(bits[x,y]) Color.BLACK else Color.WHITE)
            handler.post {
                if (tab != "Ghép đôi" || isDestroyed) return@post
                val image = ImageView(this).apply { setImageBitmap(bitmap); contentDescription = "Mã QR dùng một lần, hết hạn sau hai phút" }
                body.addView(image, LinearLayout.LayoutParams(-1,dp(300)))
                val showCode=Button(this).apply {text="Hiện mã để ghép đôi webcam máy tính";isAllCaps=false
                    setOnClickListener {
                        val content=TextView(this@MainActivity).apply{text=code;setTextIsSelectable(true);setPadding(24,24,24,24)}
                        android.app.AlertDialog.Builder(this@MainActivity).setTitle("Mã dùng một lần; giữ riêng tư")
                            .setView(ScrollView(this@MainActivity).apply{addView(content)}).setPositiveButton("Đóng",null).show()
                    }
                }
                body.addView(showCode)
                message.text = "Quét bằng máy camera. Tạo mã mới sẽ vô hiệu mã cũ."
                handler.postDelayed({ if(image.parent === body) body.removeView(image);if(showCode.parent===body) body.removeView(showCode) },120_000)
            }
        } }
        button("Máy camera: quét mã ghép đôi") {
            if(CameraData.running) { message.text="Dừng máy camera trước khi quét mã ghép đôi mới."; return@button }
            scanKind="pair"
            if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)
                scan.launch(Intent(this, PairScanActivity::class.java))
            else scanPermission.launch(Manifest.permission.CAMERA)
        }
        val paste = input("Hoặc dán mã childsafe-pair:…")
        button("Máy camera: dùng mã đã dán") { pairCamera(paste.text.toString()) }
        button("Mở chế độ máy camera") { startActivity(Intent(this,CameraActivity::class.java)) }
    }

    private fun createMonitor(address: String,fingerprint: String)=network {
        val existing=SecureConfig(this).load("monitor")
        if(existing!=null) {
            handler.post{message.text=if(existing.relayUrl==address && existing.certificatePin==fingerprint) "Máy chủ này đã được cấu hình; có thể tạo mã ghép đôi camera." else "Máy đã có cấu hình theo dõi khác. Xem phần Cài đặt trước khi thay máy chủ."}
            return@network
        }
        val room=RelayApi(address,fingerprint).call("POST","/v1/rooms")
        SecureConfig(this).save(DeviceConfig("monitor",address,room.getString("room_id"),
            room.getString("monitor_token"),EventCipher.encode(EventCipher.newKey()),certificatePin=fingerprint))
        handler.post{message.text="Đã tạo máy theo dõi. Bấm tạo mã ghép đôi cho máy camera."}
    }

    private fun pairCamera(code: String) = network {
        require(code.startsWith("childsafe-pair:") && code.length <= 8192)
        val p = JSONObject(EventCipher.decode(code.removePrefix("childsafe-pair:")).toString(Charsets.UTF_8))
        require(p.getInt("version")==1 && EventCipher.decode(p.getString("key")).size==32)
        val api = RelayApi(p.getString("relay_url"),p.optString("certificate_pin"))
        val joined = api.call("POST","/v1/join",body=JSONObject().put("invitation",p.getString("invitation")).put("name",Build.MODEL.take(60)))
        require(joined.getString("room_id")==p.getString("room_id"))
        SecureConfig(this).save(DeviceConfig("camera",p.getString("relay_url"),joined.getString("room_id"),
            joined.getString("camera_token"),p.getString("key"),joined.getString("camera_id"),p.optString("certificate_pin")))
        handler.post { message.text = "Đã ghép đôi. Bấm Mở chế độ máy camera để cấp quyền và bật phân tích." }
    }

    private fun renderSettings() {
        body.removeAllViews()
        card("Phạm vi nguyên mẫu", "Hỗ trợ chú ý khi vật cầm tay tiến sát vùng miệng. Không xác nhận nuốt hay nghẹn. Chưa kiểm chứng triển khai thực tế.")
        card("Phạm vi vật hiện tại", "Nguy cơ cao cần mask đủ tin cậy và loại vật thống nhất; detector hiện hỗ trợ dao, kéo, nĩa cho nhánh này. Bút, đũa và đồng xu chưa có model định vị riêng. Vật khác có thể chỉ tạo cảnh giác hoặc không được nhận ra.")
        card("Kết nối", "Dịch vụ truyền sự kiện cần được cấu hình riêng. Nhận cảnh báo Internet khi điện thoại ngủ cần kiểm thử thêm; chưa có thông báo đẩy FCM trong bản này.")
        button("Cài đặt âm báo và quyền thông báo") { startActivity(Intent(android.provider.Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(android.provider.Settings.EXTRA_APP_PACKAGE,packageName)) }
        button("Kiểm tra khả năng chạy mô hình trên máy") { startActivity(Intent(this,DeviceCheckActivity::class.java)) }
        button("Quản lý camera đã ghép đôi") { network {
            val c=SecureConfig(this).load("monitor")?:error("Monitor not configured")
            val api=RelayApi(c.relayUrl,c.certificatePin)
            val cameras=api.call("GET","/v1/rooms/${c.roomId}/snapshot",c.token).getJSONArray("cameras")
            val names=Array(cameras.length()){i->cameras.getJSONObject(i).getString("name")}
            handler.post {
                if(isDestroyed) return@post
                if(names.isEmpty()){message.text="Chưa có camera ghép đôi.";return@post}
                android.app.AlertDialog.Builder(this).setTitle("Chọn camera cần ngắt ghép đôi").setItems(names){_,index->
                    val camera=cameras.getJSONObject(index)
                    android.app.AlertDialog.Builder(this).setTitle("Ngắt ghép đôi ${names[index]}?")
                        .setMessage("Máy này sẽ không gửi thêm sự kiện vào phòng theo dõi. Muốn dùng lại cần ghép đôi mới.")
                        .setNegativeButton("Giữ lại",null).setPositiveButton("Ngắt ghép đôi"){_,_-> network {
                            api.call("DELETE","/v1/rooms/${c.roomId}/cameras/${camera.getString("id")}",c.token)
                            handler.post{message.text="Đã thu hồi quyền gửi của camera."}
                        }}.show()
                }.setNegativeButton("Đóng",null).show()
            }
        } }
        button("Xóa cấu hình máy theo dõi trên máy này") { stopService(Intent(this,MonitorService::class.java)); network { SecureConfig(this).clear("monitor"); handler.post { message.text="Đã xóa cấu hình cục bộ; thiết bị camera chưa bị thu hồi quyền trên relay." } } }
    }
    private fun requestMonitor() {
        if(Build.VERSION.SDK_INT>=33 && ContextCompat.checkSelfPermission(this,Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)
            permission.launch(Manifest.permission.POST_NOTIFICATIONS) else startMonitor()
    }
    private fun startMonitor() { ContextCompat.startForegroundService(this,Intent(this,MonitorService::class.java)); message.text="Đang bật theo dõi…" }
    private fun network(action: () -> Unit) {
        message.text="Đang xử lý…"
        io.execute { try { action() } catch(e: Exception) { handler.post { message.text=if(e is RelayError) "Yêu cầu không thành công (HTTP ${e.status}). Kiểm tra mã ghép đôi hoặc quyền thiết bị." else "Không hoàn tất được. Kiểm tra địa chỉ HTTPS, fingerprint và kết nối; mã ghép đôi có thể đã hết hạn." } } }
    }
    override fun onStart() { super.onStart(); resumed=true; handler.post(refresh) }
    override fun onStop() { resumed=false; handler.removeCallbacks(refresh); super.onStop() }
    override fun onDestroy() { io.shutdownNow(); super.onDestroy() }
}
