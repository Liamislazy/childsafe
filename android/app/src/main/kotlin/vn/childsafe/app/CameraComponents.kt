package vn.childsafe.app

import android.graphics.Bitmap
import org.json.JSONObject
import java.io.Closeable

/** Serial inference and event transport have independent lifetimes and own no Activity. */
interface CameraEngine : Closeable {
    val backend: String
    fun process(bitmap: Bitmap, captured: Long): VisionOutput
    fun reset()
}

interface CameraEventSink : AutoCloseable {
    val error: String
    fun submit(state: JSONObject, captured: Long, alert: Boolean)
}
