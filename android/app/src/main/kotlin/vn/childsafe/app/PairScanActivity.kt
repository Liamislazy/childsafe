package vn.childsafe.app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.zxing.BinaryBitmap
import com.google.zxing.MultiFormatReader
import com.google.zxing.PlanarYUVLuminanceSource
import com.google.zxing.common.HybridBinarizer
import java.util.concurrent.Executors

/** Offline QR decoding. A QR is data only; it never opens a URL or executes code. */
class PairScanActivity : ComponentActivity() {
    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var done = false
    private var provider: ProcessCameraProvider? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefix=if(intent.getStringExtra("kind")=="relay") "childsafe-relay:" else "childsafe-pair:"
        val previewView = PreviewView(this)
        previewView.contentDescription = "Đưa mã ghép đôi ChildSafe vào giữa khung hình"
        setContentView(previewView)
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (isDestroyed) return@addListener
            try {
                provider = future.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                val reader = MultiFormatReader()
                analysis.setAnalyzer(executor) { image ->
                    try {
                        if (done) return@setAnalyzer
                        val plane = image.planes[0]
                        val buffer = plane.buffer
                        val pixels = ByteArray(image.width * image.height)
                        for (y in 0 until image.height) for (x in 0 until image.width) {
                            val offset = y * plane.rowStride + x * plane.pixelStride
                            pixels[y * image.width + x] = buffer.get(offset)
                        }
                        val text = reader.decode(BinaryBitmap(HybridBinarizer(PlanarYUVLuminanceSource(
                            pixels,image.width,image.height,0,0,image.width,image.height,false)))).text
                        if (text.startsWith(prefix) && text.length <= 8192) {
                            done = true
                            runOnUiThread { setResult(Activity.RESULT_OK, Intent().putExtra("pairing", text)); finish() }
                        }
                    } catch (_: Exception) { /* No matching QR in this frame. */ }
                    finally { reader.reset(); image.close() }
                }
                provider!!.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (_: Exception) { setResult(Activity.RESULT_CANCELED); finish() }
        }, ContextCompat.getMainExecutor(this))
    }
    override fun onDestroy() { done = true; provider?.unbindAll(); executor.shutdown(); super.onDestroy() }
}
