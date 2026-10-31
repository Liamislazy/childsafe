package vn.childsafe.app

import android.os.SystemClock
import org.json.JSONObject
import vn.childsafe.core.EventCipher
import java.util.ArrayDeque

/** Network owns JSON only, never camera frames. Inference never waits for I/O. */
class CameraPublisher(private val config: DeviceConfig): CameraEventSink {
    private data class Pending(val state: String,val captured: Long)
    private val lock=Object()
    private val alerts=ArrayDeque<Pending>()
    private var latest: Pending?=null
    @Volatile private var stopped=false
    @Volatile override var error: String=""; private set
    @Volatile var dropped=0; private set
    private val worker=Thread({run()},"childsafe-camera-events")
    init { require(config.role=="camera"); worker.start() }
    override fun submit(state: JSONObject,captured: Long,alert: Boolean) {
        val item=Pending(state.toString(),captured)
        synchronized(lock) {
            if(stopped || SystemClock.elapsedRealtime()-captured>=2500) { dropped++; return }
            if(alert) { if(alerts.size>=4) {alerts.removeFirst(); dropped++}; alerts.addLast(item) }
            else latest=item
            lock.notifyAll()
        }
    }
    private fun run() {
        val api=try{RelayApi(config.relayUrl,config.certificatePin)}catch(_:Exception){error="Cấu hình relay chưa hợp lệ";return}
        val key=try{EventCipher.decode(config.key)}catch(_:Exception){error="Khóa ghép đôi chưa hợp lệ";return}
        var epoch=""; var sequence=0L; var remote=0L; var synced=0L
        while(!stopped) {
            val item=synchronized(lock) {
                while(!stopped && alerts.isEmpty() && latest==null) lock.wait(500)
                if(stopped) null else if(alerts.isNotEmpty()) alerts.removeFirst() else latest.also{latest=null}
            } ?:continue
            try {
                val deadline=item.captured+2500
                if(SystemClock.elapsedRealtime()>=deadline) {dropped++;continue}
                var reconnect=false
                if(epoch.isEmpty()) {
                    val start=SystemClock.elapsedRealtime()
                    val session=api.call("POST","/v1/cameras/${config.cameraId}/session",config.token)
                    synced=SystemClock.elapsedRealtime()
                    if(synced-start>2500) throw IllegalStateException("Slow handshake")
                    remote=session.getLong("server_time_ms"); epoch=session.getString("epoch"); sequence=0; reconnect=true
                }
                val now=SystemClock.elapsedRealtime()
                if(stopped || now>=deadline) {dropped++;continue}
                val state=JSONObject(item.state)
                if(reconnect && state.getString("risk")!="normal" && state.getString("observation")!="unobservable") state.put("kind","alert")
                val message=EventCipher.seal(key,state.toString().toByteArray(Charsets.UTF_8),config.roomId,config.cameraId,
                    epoch,sequence++,remote+minOf(deadline,now+2500)-synced)
                val body=JSONObject().put("version",1).put("epoch",message.epoch).put("sequence",message.sequence)
                    .put("expires_at_ms",message.expiresAtMs).put("nonce",message.nonce).put("ciphertext",message.ciphertext)
                api.call("POST","/v1/cameras/${config.cameraId}/events",config.token,body)
                error=""
            } catch(e:Exception) {
                if(e is RelayError && e.status==401) {
                    error="Camera đã bị ngắt ghép đôi; dừng camera và ghép lại nếu cần"
                    close();return
                }
                error="Chưa gửi được cảnh báo; kiểm tra mạng và quyền thiết bị"
                epoch=""; dropped++
                synchronized(lock) { if(!stopped) lock.wait(500) }
            }
        }
    }
    override fun close() { synchronized(lock){stopped=true;alerts.clear();latest=null;lock.notifyAll()} }
}
