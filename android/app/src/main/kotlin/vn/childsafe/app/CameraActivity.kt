package vn.childsafe.app

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.view.WindowManager
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject

class CameraActivity: ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var toggle: Button
    private val handler=Handler(Looper.getMainLooper())
    private var service: CameraNodeService?=null
    private var bound=false
    private var visible=false
    private val permission=registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        val camera=ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED
        val notification=Build.VERSION.SDK_INT<33 || ContextCompat.checkSelfPermission(this,Manifest.permission.POST_NOTIFICATIONS)==PackageManager.PERMISSION_GRANTED
        if(camera && notification) startCamera() else status.text="Cần quyền camera và thông báo để bật máy camera."
    }
    private val connection=object:ServiceConnection {
        override fun onServiceConnected(name: ComponentName,binder: IBinder) {service=(binder as CameraNodeService.LocalBinder).service;service?.preview(preview.surfaceProvider)}
        override fun onServiceDisconnected(name: ComponentName) {service=null;bound=false}
    }
    private val refresh=object:Runnable {
        override fun run() {
            if(!visible) return
            val state=try{JSONObject(CameraData.result)}catch(_:Exception){null}
            val age=SystemClock.elapsedRealtime()-CameraData.lastResultElapsed
            val risk=if(age>2500) "Chưa có kết quả mới" else when(state?.optString("risk")){"danger"->"NGUY CƠ CAO";"caution"->"CẦN CHÚ Ý";else->"Chưa ghi nhận nguy cơ; quan sát có giới hạn"}
            status.text="$risk\n${state?.optString("reason").orEmpty()}\n${CameraData.message}\nXử lý: ${CameraData.processingMs} ms · Người: ${CameraData.people} · Vật: ${CameraData.objects}\n${CameraData.backend}\nĐã xử lý ${CameraData.frameCount} ảnh. Tuổi ảnh đo từ lúc ứng dụng nhận ảnh, không phải độ trễ cảm biến."
            toggle.text=if(CameraData.running) "Tắt máy camera" else "Bật máy camera"
            handler.postDelayed(this,500)
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
        val root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(16,32,16,16)}
        ViewCompat.setOnApplyWindowInsetsListener(root) {view,insets ->
            val bars=insets.getInsets(WindowInsetsCompat.Type.systemBars());val gap=(16*resources.displayMetrics.density).toInt()
            view.setPadding(gap+bars.left,gap+bars.top,gap+bars.right,gap+bars.bottom);insets
        }
        root.addView(TextView(this).apply{text="ChildSafe · Máy camera";textSize=22f})
        root.addView(TextView(this).apply{text="Xử lý trên máy; không ghi video. Chỉ thử với đạo cụ an toàn.";textSize=16f})
        preview=PreviewView(this)
        root.addView(preview,LinearLayout.LayoutParams(-1,0,1f))
        status=TextView(this).apply{textSize=15f}
        root.addView(status)
        toggle=Button(this).apply{text="Bật máy camera";isAllCaps=false;setOnClickListener{
            if(CameraData.running) {
                if(bound){unbindService(connection);bound=false};service=null
                stopService(Intent(this@CameraActivity,CameraNodeService::class.java))
            }
            else {
                val needed=mutableListOf(Manifest.permission.CAMERA)
                if(Build.VERSION.SDK_INT>=33) needed.add(Manifest.permission.POST_NOTIFICATIONS)
                permission.launch(needed.toTypedArray())
            }
        }}
        root.addView(toggle)
        root.addView(Button(this).apply{text="Đổi camera trước/sau";isAllCaps=false;setOnClickListener{service?.switchCamera()}})
        root.addView(Button(this).apply{text="Gửi tín hiệu thử đến máy theo dõi";isAllCaps=false;setOnClickListener{service?.testNotification()}})
        setContentView(root)
    }
    private fun startCamera() {
        ContextCompat.startForegroundService(this,Intent(this,CameraNodeService::class.java))
        if(!bound) bound=bindService(Intent(this,CameraNodeService::class.java),connection,BIND_AUTO_CREATE)
    }
    override fun onStart() {super.onStart();visible=true;if(CameraData.running && !bound) bound=bindService(Intent(this,CameraNodeService::class.java),connection,BIND_AUTO_CREATE);handler.post(refresh)}
    override fun onStop() {visible=false;handler.removeCallbacks(refresh);service?.preview(null);if(bound){unbindService(connection);bound=false};service=null;super.onStop()}
}
