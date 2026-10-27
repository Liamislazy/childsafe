package vn.childsafe.app

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Binder
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

object CameraData {
    val generation=AtomicLong()
    @Volatile var running=false
    @Volatile var message="Camera chưa bật"
    @Volatile var processingMs=0L
    @Volatile var frameCount=0L
    @Volatile var objects=0
    @Volatile var people=0
    @Volatile var backend=""
    @Volatile var result=""
    @Volatile var lastResultElapsed=0L
}

class CameraNodeService: LifecycleService() {
    inner class LocalBinder: Binder() { val service get()=this@CameraNodeService }
    private val executor=Executors.newSingleThreadExecutor()
    private val main=Handler(Looper.getMainLooper())
    private val generation=AtomicLong()
    @Volatile private var closed=false
    private var provider: ProcessCameraProvider?=null
    private var analysis: ImageAnalysis?=null
    private var surface: Preview.SurfaceProvider?=null
    private var facing=CameraSelector.LENS_FACING_BACK
    private var engine: NativeVision?=null
    private var publisher: CameraPublisher?=null
    @Volatile private var lastFrame=0L
    @Volatile private var lastSent=0L
    @Volatile private var lastRisk="normal"
    @Volatile private var lastTrusted=0L
    private val statusId=UUID.randomUUID().toString()
    private var ownerEpoch=-1L
    private val watchdog=object:Runnable {
        override fun run() {
            if(closed) return
            if(lastFrame>0 && SystemClock.elapsedRealtime()-lastFrame>1500) {
                CameraData.message="Không nhận được ảnh mới từ camera"
                emitLimited("Camera không có ảnh mới")
            }
            main.postDelayed(this,500)
        }
    }
    override fun onBind(intent: Intent): IBinder { super.onBind(intent);return LocalBinder() }
    override fun onStartCommand(intent: Intent?,flags: Int,startId: Int): Int {
        super.onStartCommand(intent,flags,startId)
        if(intent?.action=="STOP") {stopSelf();return START_NOT_STICKY}
        if(CameraData.running) return START_NOT_STICKY
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED) {
            CameraData.message="Chưa có quyền camera";stopSelf();return START_NOT_STICKY
        }
        val manager=getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("camera","Máy camera đang hoạt động",NotificationManager.IMPORTANCE_LOW))
        val open=PendingIntent.getActivity(this,0,Intent(this,CameraActivity::class.java),PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        startForeground(2,Notification.Builder(this,"camera").setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("ChildSafe: máy camera đang hoạt động").setContentText("Phân tích trên máy; không ghi video")
            .setContentIntent(open).setOngoing(true).build())
        CameraData.running=true; CameraData.message="Đang kiểm tra và nạp mô hình…"
        ownerEpoch=CameraData.generation.incrementAndGet()
        CameraData.result="";CameraData.lastResultElapsed=0;CameraData.frameCount=0;CameraData.processingMs=0
        executor.execute {
            try {
                if(closed) return@execute
                val config=SecureConfig(this).load("camera")?:error("Camera not paired")
                publisher=CameraPublisher(config)
                engine=NativeVision(this)
                if(closed) return@execute
                CameraData.backend=engine!!.backend
                main.post {if(!closed) bindCamera()}
            } catch(_:Throwable) {
                if(closed || ownerEpoch!=CameraData.generation.get()) return@execute
                CameraData.message="Không mở được mô hình hoặc cấu hình camera; cần kiểm tra gói mô hình và ghép đôi"
                emitLimited(CameraData.message)
                main.post {stopSelf()}
            }
        }
        main.post(watchdog)
        return START_NOT_STICKY
    }
    fun preview(value: Preview.SurfaceProvider?) {surface=value;if(engine!=null && !closed) bindCamera()}
    fun switchCamera() {facing=if(facing==CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK;bindCamera()}
    fun testNotification() {
        val state=JSONObject().put("event_id",UUID.randomUUID().toString()).put("kind","test")
            .put("captured_at_ms",System.currentTimeMillis()).put("risk","normal").put("observation","unobservable")
            .put("reason","Tín hiệu thử do người dùng gửi").put("people",JSONArray())
        publisher?.submit(state,SystemClock.elapsedRealtime(),true)
    }
    @Suppress("DEPRECATION")
    private fun bindCamera() {
        if(closed || engine==null) return
        val epoch=generation.incrementAndGet()
        analysis?.clearAnalyzer();provider?.unbindAll()
        val future=ProcessCameraProvider.getInstance(this)
        executor.execute {
            engine?.reset()
            main.post {
                future.addListener({
                    if(closed || epoch!=generation.get()) return@addListener
                    try {
                        provider=future.get()
                        val selector=CameraSelector.Builder().requireLensFacing(facing).build()
                        if(!provider!!.hasCamera(selector)) error("Lens unavailable")
                        val analyzer=ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setTargetResolution(Size(1280,720)).build()
                        analysis=analyzer
                        analyzer.setAnalyzer(executor) {image->
                            var bitmap: Bitmap?=null
                            val arrived=SystemClock.elapsedRealtime()
                            lastFrame=arrived
                            try {
                                if(closed || epoch!=generation.get()) return@setAnalyzer
                                val original=image.toBitmap()
                                bitmap=if(image.imageInfo.rotationDegrees!=0) {
                                    val matrix=Matrix().apply{postRotate(image.imageInfo.rotationDegrees.toFloat())}
                                    Bitmap.createBitmap(original,0,0,original.width,original.height,matrix,true).also{if(it!==original)original.recycle()}
                                } else original
                                val result=engine!!.process(bitmap!!,arrived)
                                if(closed || epoch!=generation.get()) return@setAnalyzer
                                CameraData.processingMs=SystemClock.elapsedRealtime()-arrived
                                CameraData.frameCount++
                                if(CameraData.processingMs>250) {
                                    engine!!.reset()
                                    CameraData.message="Xử lý quá 250 ms; kết quả này không tạo cảnh báo mới"
                                    emitLimited(CameraData.message)
                                } else {
                                    lastRisk=result.state.getString("risk");lastTrusted=SystemClock.elapsedRealtime()
                                    CameraData.result=result.state.toString();CameraData.lastResultElapsed=lastTrusted
                                    CameraData.people=result.people;CameraData.objects=result.objects
                                    CameraData.message=publisher?.error?.takeIf{it.isNotBlank()}?:"Đang phân tích; ngưỡng Android chưa được kiểm chứng"
                                    val now=SystemClock.elapsedRealtime()
                                    if(result.notify || now-lastSent>=1000) { publisher?.submit(result.state,arrived,result.notify);lastSent=now }
                                }
                            } catch(_:Exception) {
                                engine?.reset();CameraData.message="Không phân tích được ảnh hiện tại";emitLimited(CameraData.message)
                            } finally {bitmap?.recycle();image.close()}
                        }
                        if(surface!=null) {
                            val preview=Preview.Builder().build().also{it.setSurfaceProvider(surface)}
                            provider!!.bindToLifecycle(this,selector,analyzer,preview)
                        } else provider!!.bindToLifecycle(this,selector,analyzer)
                    } catch(_:Exception) {CameraData.message="Không mở được camera; có thể đang bị ứng dụng khác sử dụng";emitLimited(CameraData.message)}
                },ContextCompat.getMainExecutor(this))
            }
        }
    }
    @Synchronized private fun emitLimited(reason: String) {
        val now=SystemClock.elapsedRealtime()
        if(now-lastSent<1000) return
        val risk=if(now-lastTrusted<=800) lastRisk else "normal"
        val state=JSONObject().put("event_id",statusId).put("kind","status").put("captured_at_ms",System.currentTimeMillis())
            .put("risk",risk).put("observation","unobservable").put("reason",reason.take(1000)).put("people",JSONArray())
        publisher?.submit(state,now,false);lastSent=now
        CameraData.result=state.toString();CameraData.lastResultElapsed=now
    }
    override fun onDestroy() {
        closed=true;generation.incrementAndGet()
        if(CameraData.generation.compareAndSet(ownerEpoch,ownerEpoch+1)) CameraData.running=false
        main.removeCallbacks(watchdog);analysis?.clearAnalyzer();provider?.unbindAll()
        publisher?.close()
        executor.execute {engine?.close();engine=null;publisher?.close();publisher=null}
        executor.shutdown()
        super.onDestroy()
    }
}
