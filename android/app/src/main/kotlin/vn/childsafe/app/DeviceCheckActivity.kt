package vn.childsafe.app

import android.os.Build
import android.os.Bundle
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

/** Numeric diagnostics only: no camera, microphone, connection tokens or images. */
class DeviceCheckActivity: ComponentActivity() {
    private val worker=Executors.newSingleThreadExecutor()
    private lateinit var output: TextView
    private lateinit var run: Button
    private var report: JSONObject?=null
    private val export=registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if(uri!=null && report!=null) contentResolver.openOutputStream(uri)?.use{it.write(report!!.toString(2).toByteArray())}
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(24,48,24,24)}
        ViewCompat.setOnApplyWindowInsetsListener(root){view,insets->
            val bars=insets.getInsets(WindowInsetsCompat.Type.systemBars());val gap=(16*resources.displayMetrics.density).toInt()
            view.setPadding(gap+bars.left,gap+bars.top,gap+bars.right,gap+bars.bottom);insets
        }
        root.addView(TextView(this).apply{text="ChildSafe · Kiểm tra thiết bị";textSize=22f})
        root.addView(TextView(this).apply{text="Kiểm tra mô hình bằng dữ liệu tổng hợp, không mở camera. Kết quả này không chứng minh độ chính xác nhận diện hay FPS thực tế.";textSize=16f})
        output=TextView(this).apply{textSize=15f;setTextIsSelectable(true)}
        root.addView(ScrollView(this).apply{addView(output)},LinearLayout.LayoutParams(-1,0,1f))
        run=Button(this).apply{text="Chạy kiểm tra trên máy này";isAllCaps=false;setOnClickListener{check()}}
        root.addView(run)
        root.addView(Button(this).apply{text="Xuất số liệu JSON";isAllCaps=false;setOnClickListener{if(report!=null) export.launch("childsafe-device-check.json")}})
        setContentView(root)
    }
    private fun check() {
        if(CameraData.running){output.text="Dừng máy camera trước khi chạy phép đo này.";return}
        run.isEnabled=false;output.text="Đang nạp và kiểm tra mô hình…"
        worker.execute {
            val result=JSONObject().put("device",Build.MODEL).put("manufacturer",Build.MANUFACTURER)
                .put("android",Build.VERSION.RELEASE).put("api",Build.VERSION.SDK_INT)
                .put("created_at_ms",System.currentTimeMillis())
                .put("abis",org.json.JSONArray(Build.SUPPORTED_ABIS.toList()))
                .put("schema_version",1)
                .put("app_version",packageManager.getPackageInfo(packageName,0).versionName)
            try { NativeVision(this).use{engine->result.put("checks",engine.selfCheck())} }
            catch(e:Throwable){result.put("numerical_check_passed",false).put("error_type",e.javaClass.simpleName)
                .put("error_message",e.message.orEmpty().take(2000))
                .put("stack",org.json.JSONArray(e.stackTrace.take(8).map{it.toString()}))
                .put("note","Không hoàn tất kiểm tra; cần đối chiếu gói mô hình và hỗ trợ thiết bị")}
            File(filesDir,"device-check.json").writeText(result.toString(2))
            report=result
            runOnUiThread{if(!isDestroyed){output.text=result.toString(2);run.isEnabled=true}}
        }
    }
    override fun onDestroy(){worker.shutdown();super.onDestroy()}
}
