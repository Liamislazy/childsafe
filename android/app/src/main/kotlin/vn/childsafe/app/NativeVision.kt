package vn.childsafe.app

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.json.JSONArray
import org.json.JSONObject
import vn.childsafe.core.*
import java.io.File
import java.nio.FloatBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.UUID
import kotlin.math.*

data class VisionOutput(val state: JSONObject,val notify: Boolean,val processingMs: Long,val people: Int,val objects: Int)

/** All inference calls and close() must use the same serial camera executor. */
class NativeVision(private val context: Context): CameraEngine {
    private val manifest=JSONObject(context.assets.open("models/manifest.json").bufferedReader().use{it.readText()})
    private val config=manifest.getJSONObject("runtime_config")
    private val qc=config.getJSONObject("quality")
    private val temporal=config.getJSONObject("temporal")
    private val rules=config.getJSONObject("rules")
    private val environment=OrtEnvironment.getEnvironment()
    private val modelSettings=manifest.getJSONObject("classifier")
    private val detectorSettings=manifest.getJSONObject("detector")
    private val inputSize=detectorSettings.getInt("input_size")
    private val labels=detectorSettings.getJSONObject("classes").let { j -> (0 until j.length()).map{j.getString(it.toString())} }
    private val classes=modelSettings.getJSONArray("classes").let { j -> (0 until j.length()).map{j.getString(it)} }
    private val unknown=modelSettings.getJSONArray("unknown_classes").let { j -> (0 until j.length()).map{j.getString(it)}.toSet() }
    private val tracker=PersonTracker()
    private val people=mutableMapOf<Int,History>()
    private val instance=UUID.randomUUID().toString()
    private val resources=mutableListOf<()->Unit>()
    private var previousSize: Pair<Int,Int>?=null
    private val detector: OrtSession
    private val classifier: OrtSession
    private val face: FaceLandmarker
    private val hand: HandLandmarker
    override val backend: String

    private class History(val gate: CameraRiskGate) {
        var previousMouth: Point?=null; var mouthTime=0L
        var mask: VisibleMask?=null; var maskSince=0L; var maskTime=0L
        var objectBox: Box?=null; var objectLabel=""
        var classification: Pair<String,Double>?=null; var classTime=0L
        var near=false;var strongNear=false
    }
    private data class Pose(val track: PersonTrack,val face: List<Point>,val hands: MutableList<List<Point>>,
                            val mouth: List<Point>,val faceWidth: Double,val visible: Boolean,val reason: String)
    init {
        try {
        require(manifest.getInt("schema_version")==1)
        listOf("classifier","detector","face","hand").forEach(::verifiedFile)
        val detectorFile=verifiedFile("detector")
        var nnapi=true
        detector=try { session(detectorFile,true) } catch (_: OrtException) { nnapi=false; session(detectorFile,false) }
        resources.add{detector.close()}
        classifier=try { session(verifiedFile("classifier"),nnapi) } catch (_: OrtException) { session(verifiedFile("classifier"),false) }
        resources.add{classifier.close()}
        backend=if(nnapi) "ONNX NNAPI (có thể có CPU fallback)" else "ONNX CPU"
        face=FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("models/"+manifest.getJSONObject("face").getString("file")).build())
            .setRunningMode(RunningMode.IMAGE).setNumFaces(2).setOutputFacialTransformationMatrixes(true)
            .setMinFaceDetectionConfidence(.5f).setMinFacePresenceConfidence(.5f).build())
        resources.add{face.close()}
        hand=HandLandmarker.createFromOptions(context,HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("models/"+manifest.getJSONObject("hand").getString("file")).build())
            .setRunningMode(RunningMode.IMAGE).setNumHands(2).setMinHandDetectionConfidence(.5f)
            .setMinHandPresenceConfidence(.5f).build())
        resources.add{hand.close()}
        } catch(t: Throwable) { resources.asReversed().forEach{try{it()}catch(_:Exception){}}; throw t }
    }

    private fun verifiedFile(role: String): File {
        val j=manifest.getJSONObject(role); val name=j.getString("file")
        require(File(name).name==name && '/' !in name && '\\' !in name)
        val expected=j.getString("sha256")
        val data=context.assets.open("models/$name").use{it.readBytes()}
        require(MessageDigest.getInstance("SHA-256").digest(data).hex()==expected) { "Model checksum mismatch" }
        val dir=File(context.cacheDir,"models").apply{mkdirs()}
        return File(dir,"${expected.take(16)}-$name").also { file ->
            if(!file.isFile || MessageDigest.getInstance("SHA-256").digest(file.readBytes()).hex()!=expected) {
                val temp=File(dir,file.name+".tmp"); temp.writeBytes(data)
                check(temp.renameTo(file)) { "Could not cache model" }
            }
        }
    }
    private fun ByteArray.hex()=joinToString(""){"%02x".format(it.toInt() and 255)}
    private fun session(file: File, nnapi: Boolean): OrtSession = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(2); options.setInterOpNumThreads(1)
        if(nnapi) options.addNnapi()
        environment.createSession(file.absolutePath,options)
    }

    private fun detectorInput(bitmap: Bitmap,m: Letterbox): FloatArray {
        val resized=Bitmap.createScaledBitmap(bitmap,m.resizedWidth,m.resizedHeight,true)
        val padded=Bitmap.createBitmap(inputSize,inputSize,Bitmap.Config.ARGB_8888)
        Canvas(padded).apply { drawColor(Color.rgb(114,114,114)); drawBitmap(resized,m.left.toFloat(),m.top.toFloat(),Paint()) }
        val out=rgbTensor(padded,null,null)
        if(resized!==bitmap) resized.recycle()
        padded.recycle()
        return out
    }
    private fun rgbTensor(bitmap: Bitmap,mean: DoubleArray?,std: DoubleArray?): FloatArray {
        val pixels=IntArray(bitmap.width*bitmap.height)
        bitmap.getPixels(pixels,0,bitmap.width,0,0,bitmap.width,bitmap.height)
        val out=FloatArray(pixels.size*3)
        for(i in pixels.indices) {
            val p=pixels[i]
            out[i]=((((p ushr 16) and 255)/255.0-(mean?.get(0)?:0.0))/(std?.get(0)?:1.0)).toFloat()
            out[pixels.size+i]=((((p ushr 8) and 255)/255.0-(mean?.get(1)?:0.0))/(std?.get(1)?:1.0)).toFloat()
            out[2*pixels.size+i]=(((p and 255)/255.0-(mean?.get(2)?:0.0))/(std?.get(2)?:1.0)).toFloat()
        }
        return out
    }
    private fun detect(bitmap: Bitmap,mapping: Letterbox): Triple<List<ObjectDetection>,FloatArray,Pair<Int,Int>> {
        val tensor=OnnxTensor.createTensor(environment,FloatBuffer.wrap(detectorInput(bitmap,mapping)),longArrayOf(1,3,inputSize.toLong(),inputSize.toLong()))
        tensor.use { input -> detector.run(mapOf(detector.inputNames.single() to input)).use { result ->
            val values=detector.outputNames.map { result.get(it).get() as OnnxTensor }
            val prediction=values.single{it.info.shape.size==3}
            val proto=values.single{it.info.shape.size==4}
            val raw=prediction.floatBuffer.let{b->FloatArray(b.remaining()).also{b.get(it)}}
            val masks=proto.floatBuffer.let{b->FloatArray(b.remaining()).also{b.get(it)}}
            return Triple(decodeYolo(raw,prediction.info.shape[2].toInt(),labels,mapping),masks,
                proto.info.shape[3].toInt() to proto.info.shape[2].toInt())
        } }
    }

    private fun landmarks(bitmap: Bitmap,track: PersonTrack,h: History,now: Long): Pose {
        val b=track.box.clip(bitmap.width,bitmap.height)
        val x=b.left.toInt(); val y=b.top.toInt()
        val w=ceil(b.right).toInt()-x; val ht=ceil(b.bottom).toInt()-y
        if(w<2 || ht<2) return Pose(track,emptyList(),mutableListOf(),emptyList(),0.0,false,"Vùng người quá nhỏ")
        val crop=Bitmap.createBitmap(bitmap,x,y,w,ht)
        val image=BitmapImageBuilder(crop).build()
        try {
            val f=face.detect(image); val hands=hand.detect(image).landmarks().map { landmarks ->
                landmarks.map{Point(x+it.x()*w.toDouble(),y+it.y()*ht.toDouble())}
            }.toMutableList()
            val landmarks=f.faceLandmarks().singleOrNull()
                ?:return Pose(track,emptyList(),hands,emptyList(),0.0,false,"Chưa thấy rõ một khuôn mặt trong vùng người")
            val points=landmarks.map{Point(x+it.x()*w.toDouble(),y+it.y()*ht.toDouble())}
            if(points.any{!it.x.isFinite() || !it.y.isFinite()})
                return Pose(track,points,hands,emptyList(),0.0,false,"Mốc khuôn mặt không hợp lệ")
            val mouth=listOf(78,13,308,14).map{points[it]}
            val width=points[234].distance(points[454])
            val errors=mutableListOf<String>()
            if(width<qc.getDouble("min_face_px")) errors.add("Mặt quá nhỏ")
            if(Box.around(mouth).width<qc.getDouble("min_mouth_px")) errors.add("Vùng miệng quá nhỏ")
            val matrix=f.facialTransformationMatrixes().orElse(emptyList()).singleOrNull()
            if(matrix==null || matrix.size!=16 || matrix.any{!it.isFinite()}) errors.add("Chưa đo được hướng mặt")
            else {
                // Java MediaPipe uses a flat COLUMN-major matrix (not Python's row-major ndarray).
                val n0=sqrt(matrix[0]*matrix[0]+matrix[1]*matrix[1]+matrix[2]*matrix[2]).toDouble().coerceAtLeast(1e-9)
                val n1=sqrt(matrix[4]*matrix[4]+matrix[5]*matrix[5]+matrix[6]*matrix[6]).toDouble().coerceAtLeast(1e-9)
                val n2=sqrt(matrix[8]*matrix[8]+matrix[9]*matrix[9]+matrix[10]*matrix[10]).toDouble().coerceAtLeast(1e-9)
                val yaw=Math.toDegrees(atan2(-matrix[2]/n0,hypot(matrix[0]/n0,matrix[1]/n0)))
                val pitch=Math.toDegrees(atan2(matrix[6]/n1,matrix[10]/n2))
                if(abs(yaw)>qc.getDouble("max_yaw_deg") || abs(pitch)>qc.getDouble("max_pitch_deg")) errors.add("Hướng mặt che vùng miệng")
            }
            val center=Point(mouth.map{it.x}.average(),mouth.map{it.y}.average())
            if(h.previousMouth!=null && now-h.mouthTime in 1..250 && center.distance(h.previousMouth!!)/max(width,1.0)>qc.getDouble("max_landmark_jump")) errors.add("Mốc miệng chưa ổn định")
            h.previousMouth=center; h.mouthTime=now
            val faceBox=Box.around(points).clip(bitmap.width,bitmap.height)
            val quality=imageQuality(bitmap,faceBox)
            if(quality.first !in qc.getDouble("min_brightness")..qc.getDouble("max_brightness")) errors.add("Ánh sáng chưa đủ")
            if(quality.second<qc.getDouble("min_sharpness")) errors.add("Ảnh bị nhòe")
            for(hp in hands) {
                val palm=convexHull(listOf(0,1,5,9,13,17).map{hp[it]})
                if(palm.size>=3 && mouth.all{pointRegionDistance(it,palm)==0.0}) errors.add("Bàn tay che vùng miệng trên ảnh")
            }
            return Pose(track,points,hands,mouth,width,errors.isEmpty(),errors.joinToString("; "))
        } finally { image.close(); if(crop!==bitmap) crop.recycle() }
    }

    private fun imageQuality(bitmap: Bitmap,box: Box): Pair<Double,Double> {
        val x=box.left.toInt(); val y=box.top.toInt()
        val w=box.right.toInt()-x; val h=box.bottom.toInt()-y
        if(w<3 || h<3) return 0.0 to 0.0
        val pixels=IntArray(w*h); bitmap.getPixels(pixels,0,w,x,y,w,h)
        val gray=DoubleArray(pixels.size){i-> val p=pixels[i]; (((p ushr 16) and 255)+((p ushr 8) and 255)+(p and 255))/3.0 }
        var sum=0.0; var sum2=0.0; var count=0
        for(row in 1 until h-1) for(col in 1 until w-1) {
            val i=row*w+col; val l=4*gray[i]-gray[i-1]-gray[i+1]-gray[i-w]-gray[i+w]
            sum+=l; sum2+=l*l; count++
        }
        return gray.average() to max(0.0,sum2/count-(sum/count).pow(2))
    }

    private fun classify(bitmap: Bitmap,points: List<Point>): Pair<String,Double> {
        val box=Box.around(points); val side=max(box.width,box.height)*2.2
        val b=Box(box.center.x-side/2,box.center.y-side/2,box.center.x+side/2,box.center.y+side/2).clip(bitmap.width,bitmap.height)
        val x=b.left.toInt(); val y=b.top.toInt(); val w=b.right.toInt()-x; val h=b.bottom.toInt()-y
        require(w>8 && h>8)
        val crop=Bitmap.createBitmap(bitmap,x,y,w,h)
        val size=modelSettings.getInt("img_size")
        val preprocess=modelSettings.getString("preprocessing")
        val square: Bitmap
        if(preprocess=="letterbox") {
            square=Bitmap.createBitmap(size,size,Bitmap.Config.ARGB_8888)
            val ratio=min(size.toDouble()/w,size.toDouble()/h)
            val rw=round(w*ratio).toInt().coerceAtLeast(1); val rh=round(h*ratio).toInt().coerceAtLeast(1)
            Canvas(square).apply { drawColor(Color.rgb(124,116,104)); drawBitmap(crop,null,Rect((size-rw)/2,(size-rh)/2,(size-rw)/2+rw,(size-rh)/2+rh),Paint(Paint.FILTER_BITMAP_FLAG)) }
        } else {
            require(preprocess=="center_crop")
            val short=modelSettings.getInt("resize_short_edge")
            val rw=if(w<=h) short else (short*w.toDouble()/h).toInt()
            val rh=if(h<=w) short else (short*h.toDouble()/w).toInt()
            val resized=Bitmap.createScaledBitmap(crop,rw,rh,true)
            square=Bitmap.createBitmap(resized,round((rw-size)/2.0).toInt(),round((rh-size)/2.0).toInt(),size,size)
            if(square!==resized && resized!==crop) resized.recycle()
        }
        val mean=modelSettings.getJSONArray("mean").let{j->DoubleArray(3){j.getDouble(it)}}
        val std=modelSettings.getJSONArray("std").let{j->DoubleArray(3){j.getDouble(it)}}
        val input=rgbTensor(square,mean,std)
        if(square!==crop) square.recycle()
        if(crop!==bitmap) crop.recycle()
        OnnxTensor.createTensor(environment,FloatBuffer.wrap(input),longArrayOf(1,3,size.toLong(),size.toLong())).use { tensor ->
            classifier.run(mapOf(classifier.inputNames.single() to tensor)).use { result ->
                val values=(result[0] as OnnxTensor).floatBuffer.let{b->FloatArray(b.remaining()).also{b.get(it)}}
                require(values.size==classes.size && values.all{it.isFinite()})
                val temperature=modelSettings.getDouble("temperature")
                val top=values.indices.maxBy{values[it]}
                val weights=values.map{exp((it-values[top])/temperature)}
                return classes[top] to weights[top]/weights.sum()
            }
        }
    }

    private fun newHistory()=History(CameraRiskGate(maxGapMs=(temporal.getDouble("max_gap_s")*1000).toLong(),
        holdMs=(config.getJSONObject("smoothing").getDouble("lost_hold_s")*1000).toLong(),
        cautionMs=(rules.getDouble("caution_evidence_s")*1000).toLong(),dangerMs=(rules.getDouble("dwell_danger_s")*1000).toLong(),
        cooldownMs=(temporal.getDouble("cooldown_s")*1000).toLong(),calibrated=false))

    override fun process(bitmap: Bitmap,captured: Long): VisionOutput {
        val started=SystemClock.elapsedRealtime()
        if(previousSize!=null && previousSize!=(bitmap.width to bitmap.height)) reset()
        previousSize=bitmap.width to bitmap.height
        val mapping=Letterbox(bitmap.width,bitmap.height,inputSize)
        val (detections,proto,protoSize)=detect(bitmap,mapping)
        val (tracks,retired)=tracker.update(detections,captured)
        retired.forEach(people::remove)
        val poses=tracks.map { t ->
            if(t.historyReset) people.remove(t.id)
            landmarks(bitmap,t,people.getOrPut(t.id){newHistory()},captured)
        }
        val ambiguous=tracks.filter{it.ambiguous}.map{it.id}.toMutableSet()
        resolveHands(poses,ambiguous)
        val objects=detections.filter{it.label!="person"}
        val ownership=objects.associateWith { d ->
            val possible=poses.filter { p -> p.hands.any { h -> val extent=Box.around(h).let{max(it.width,it.height)}
                extent>=qc.getDouble("min_hand_px") && h.minOf{d.box.distance(it)}<=1.5*extent } }
            if(possible.size>1) ambiguous.addAll(possible.map{it.track.id})
            possible.singleOrNull()?.track?.id
        }
        val rows=JSONArray(); val signatures=mutableListOf<String>(); var level=Risk.NORMAL; var notify=false; var aggregateReason="Chưa thấy người trong khung hình"
        for(p in poses) {
            val h=people.getValue(p.track.id)
            var reason=p.reason; var local=false; var near=false; var reliable=false; var stable=false; var hazard=false; var conflict=false; var departure=false;var benign=false
            if(p.visible && p.track.id !in ambiguous) {
                val candidates=objects.filter{ownership[it]==p.track.id}.map { d ->
                    val mask=decodeMask(proto,protoSize.first,protoSize.second,d,mapping)
                    val distance=mask.distance(p.mouth)
                    val boxDistance=p.mouth.minOf{d.box.distance(it)}
                    Triple(d,mask,if(distance.isFinite()) distance else boxDistance)
                }
                val chosen=candidates.minByOrNull{it.third}
                if(chosen!=null) {
                    val (d,mask,distance)=chosen
                    val different=h.objectBox!=null && (h.objectLabel!=d.label || h.objectBox!!.iou(d.box)<.2)
                    if(different) { h.classification=null; h.mask=null;h.near=false;h.strongNear=false; h.gate.update(MouthEvidence(true,true,false,false,false,false,false,false,ambiguousPerson=true),captured) }
                    h.objectBox=d.box; h.objectLabel=d.label
                    val closestHand=p.hands.minBy{points->points.minOf{d.box.distance(it)}}
                    val handArea=Box.around(closestHand).area.coerceAtLeast(1.0)
                    local=true
                    if(distance/p.faceWidth<temporal.getDouble("d_enter")) h.near=true
                    else if(distance/p.faceWidth>temporal.getDouble("d_exit")) h.near=false
                    near=h.near
                    val geometric=mask.pixelArea>=qc.getDouble("min_mask_px") && mask.dominantFraction()>=qc.getDouble("min_mask_component_fraction")
                    reliable=geometric && d.confidence>=qc.getDouble("min_mask_conf") && mask.pixelArea/handArea<=qc.getDouble("max_hand_object_ratio")
                    if(reliable && h.mask!=null && captured-h.maskTime in 1..250 && mask.iou(h.mask!!)>=qc.getDouble("min_mask_shape_iou")) {
                        stable=captured-h.maskSince>=(qc.getDouble("mask_stability_s")*1000).toLong()
                    } else h.maskSince=captured
                    h.mask=if(reliable) mask else null; h.maskTime=captured
                    if(near && (h.classification==null || captured-h.classTime>=300)) {
                        try { h.classification=classify(bitmap,closestHand); h.classTime=captured }
                        catch (_: Exception) { h.classification=null; reason="Classifier lỗi; chế độ nhận diện hạn chế" }
                    }
                    val c=h.classification
                    val classKnown=c!=null && c.second>=config.getJSONObject("calibration").getDouble("reject_threshold") && c.first !in unknown
                    val canonical=when(d.label){"apple","banana","orange"->"fruit";"teddy bear","sports ball"->if(d.label=="teddy bear") "toy" else "ball";else->d.label}
                    conflict=classKnown && c!!.first!=canonical
                    hazard=config.getJSONObject("hazard_policy").getJSONObject("mouth_hazards").has(d.label) &&
                        d.confidence>=config.getJSONObject("hazard_policy").getDouble("detector_min_conf")
                    benign=classKnown && !conflict && !hazard && modelSettings.getJSONObject("binary_map").optString(c!!.first)=="safe"
                    val upper=(distance+mask.uncertaintyPx)/p.faceWidth
                    if(!reliable || upper>temporal.getDouble("d_exit")) h.strongNear=false
                    else if(upper<temporal.getDouble("d_enter")) h.strongNear=true
                    reliable=reliable && h.strongNear
                    departure=geometric && distance/p.faceWidth>temporal.getDouble("d_exit")
                    reason=listOf(reason,"Vật định vị: ${d.label}; phân loại: ${if(classKnown) c!!.first else "chưa chắc chắn"}").filter{it.isNotBlank()}.joinToString(". ")
                }
            }
            val elapsed=SystemClock.elapsedRealtime()-captured
            val timely=elapsed<=250
            val e=MouthEvidence(p.visible,timely,local,local,near,reliable,stable,hazard,conflict,p.track.id in ambiguous,departure,benign)
            val decision=h.gate.update(e,captured)
            signatures.add("${p.track.id}:${decision.episode}:${decision.risk.name}")
            val explanation=listOf(decision.reason,reason,if(!timely) "Xử lý vượt 250 ms; không dùng kết quả này tạo bằng chứng mới" else "").filter{it.isNotBlank()}.joinToString(". ")
            rows.put(JSONObject().put("person_id",p.track.id).put("risk",decision.risk.name.lowercase())
                .put("observation",decision.observation.name.lowercase()).put("reason",explanation.take(1000)))
            if(decision.risk.ordinal>=level.ordinal) { level=decision.risk; aggregateReason=explanation }
            notify=notify || decision.notify
        }
        // A disappeared track ends because observation was lost, not because it became safe.
        val ids=tracks.map{it.id}.toSet()
        people.keys.filter{it !in ids}.toList().forEach { id ->
            val decision=people.getValue(id).gate.update(MouthEvidence(false,false,false,false,false,false,false,false),captured)
            if(decision.risk.ordinal>level.ordinal) { level=decision.risk; aggregateReason=decision.reason }
        }
        val eventId=MessageDigest.getInstance("SHA-256").digest((instance+signatures.joinToString()+level.name).toByteArray()).hex()
        val state=JSONObject().put("event_id",eventId)
            .put("kind",if(notify) "alert" else "status").put("captured_at_ms",System.currentTimeMillis()-(SystemClock.elapsedRealtime()-captured))
            .put("risk",level.name.lowercase()).put("observation",if(poses.isEmpty() || poses.none{it.visible}) "unobservable" else "degraded")
            .put("reason",aggregateReason.take(1000)).put("people",rows)
        return VisionOutput(state,notify,SystemClock.elapsedRealtime()-started,poses.size,objects.size)
    }

    private fun resolveHands(poses: List<Pose>,ambiguous: MutableSet<Int>) {
        fun distance(p: Pose,h: List<Point>): Double? = if(p.mouth.isEmpty()) null else h.minOf{pointRegionDistance(it,p.mouth)}/max(p.faceWidth,1.0)
        val drop=poses.associate{it.track.id to mutableSetOf<Int>()}
        for(i in poses.indices) for(j in i+1 until poses.size) {
            val a=poses[i]; val b=poses[j]
            for(ia in a.hands.indices) for(ib in b.hands.indices) {
                val ha=a.hands[ia]; val hb=b.hands[ib]
                val extent=Box.around(ha).let{max(it.width,it.height)}
                if(ha.zip(hb).map{it.first.distance(it.second)}.average()>.25*extent) continue
                val da=distance(a,ha); val db=distance(b,hb)
                if(da!=null && (db==null || da*1.5<=db)) drop.getValue(b.track.id).add(ib)
                else if(db!=null && (da==null || db*1.5<=da)) drop.getValue(a.track.id).add(ia)
                else { ambiguous.add(a.track.id); ambiguous.add(b.track.id) }
            }
        }
        poses.forEach { p -> val keep=p.hands.filterIndexed{i,_->i !in drop.getValue(p.track.id)}; p.hands.clear(); p.hands.addAll(keep) }
        poses.forEach { p -> p.hands.forEach { h ->
            val center=Box.around(h).center
            poses.filter{it.track.id!=p.track.id && it.track.box.contains(center)}.forEach { other ->
                val own=distance(p,h); val theirs=distance(other,h)
                if(own==null || !(if(theirs==null) own<=1.0 else own*1.5<=theirs)) { ambiguous.add(p.track.id); ambiguous.add(other.track.id) }
            }
        } }
    }
    override fun reset() { tracker.reset(); people.clear(); previousSize=null }
    fun selfCheck(): JSONObject {
        fun input(role: String): FloatArray {
            val j=manifest.getJSONObject(role).getJSONObject("verification")
            val name=j.getString("input_file")
            require(File(name).name==name && '/' !in name && '\\' !in name)
            val data=context.assets.open("models/$name").use{it.readBytes()}
            require(MessageDigest.getInstance("SHA-256").digest(data).hex()==j.getString("input_sha256"))
            val buffer=ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also{buffer.get(it)}
        }
        val report=JSONObject().put("backend",backend).put("detector_size",inputSize)
            .put("classifier_sha256",modelSettings.getString("sha256"))
            .put("detector_sha256",detectorSettings.getString("sha256"))
        val classifierInput=input("classifier")
        val size=modelSettings.getInt("img_size")
        val start=SystemClock.elapsedRealtime()
        OnnxTensor.createTensor(environment,FloatBuffer.wrap(classifierInput),longArrayOf(1,3,size.toLong(),size.toLong())).use { t ->
            classifier.run(mapOf(classifier.inputNames.single() to t)).use { result ->
                val actual=(result[0] as OnnxTensor).floatBuffer
                val expected=modelSettings.getJSONObject("verification").getJSONArray("expected_logits")
                require(actual.remaining()==expected.length())
                var error=0.0
                for(i in 0 until expected.length()) {
                    val value=actual.get().toDouble();val target=expected.getDouble(i)
                    require(value.isFinite() && abs(value-target)<=1e-3+abs(target)*1e-3){"Classifier numerical check failed"}
                    error=max(error,abs(value-target))
                }
                report.put("classifier_max_abs_error",error)
            }
        }
        report.put("classifier_ms",SystemClock.elapsedRealtime()-start)
        val detectorInput=input("detector")
        val outputChecks=detectorSettings.getJSONObject("verification").getJSONArray("outputs")
        val times=JSONArray()
        repeat(3) {
            val before=SystemClock.elapsedRealtime()
            OnnxTensor.createTensor(environment,FloatBuffer.wrap(detectorInput),longArrayOf(1,3,inputSize.toLong(),inputSize.toLong())).use { t ->
                detector.run(mapOf(detector.inputNames.single() to t)).use { result ->
                    val tensors=detector.outputNames.map{n->result.get(n).get() as OnnxTensor}.sortedBy{v->v.info.shape.size}
                    tensors.forEachIndexed { i,value ->
                        val check=outputChecks.getJSONObject(i)
                        val shape=check.getJSONArray("shape")
                        require(value.info.shape.contentEquals(LongArray(shape.length()){k->shape.getLong(k)}))
                        val indices=check.getJSONArray("sample_indices");val expected=check.getJSONArray("expected")
                        val buffer=value.floatBuffer
                        for(k in 0 until indices.length()) {
                            val actual=buffer.get(indices.getInt(k)).toDouble();val target=expected.getDouble(k)
                            require(actual.isFinite() && abs(actual-target)<=1e-3+abs(target)*1e-3){"Detector numerical check failed"}
                        }
                    }
                }
            }
            times.put(SystemClock.elapsedRealtime()-before)
        }
        report.put("detector_ms",times)
        val bitmap=Bitmap.createBitmap(320,320,Bitmap.Config.ARGB_8888).apply{eraseColor(Color.GRAY)}
        val image=BitmapImageBuilder(bitmap).build()
        try {
            val beforeFace=SystemClock.elapsedRealtime();val f=face.detect(image)
            report.put("face_ms",SystemClock.elapsedRealtime()-beforeFace).put("synthetic_face_count",f.faceLandmarks().size)
            val beforeHand=SystemClock.elapsedRealtime();val h=hand.detect(image)
            report.put("hand_ms",SystemClock.elapsedRealtime()-beforeHand).put("synthetic_hand_count",h.landmarks().size)
        } finally {image.close();bitmap.recycle()}
        return report.put("numerical_check_passed",true).put("camera_accuracy_validated",false)
            .put("note","Thời gian chạy mô hình trên đầu vào tổng hợp; không phải FPS camera hay độ trễ cảnh báo")
    }
    override fun close() { resources.asReversed().forEach{try{it()}catch(_:Exception){}}; resources.clear() }
}
