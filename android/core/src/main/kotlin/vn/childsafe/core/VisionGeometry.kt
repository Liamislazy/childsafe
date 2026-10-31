package vn.childsafe.core

import kotlin.math.*

data class Point(val x: Double, val y: Double) {
    fun distance(other: Point) = hypot(x-other.x,y-other.y)
}
data class Box(val left: Double,val top: Double,val right: Double,val bottom: Double) {
    val width get() = max(0.0,right-left)
    val height get() = max(0.0,bottom-top)
    val area get() = width*height
    val center get() = Point((left+right)/2,(top+bottom)/2)
    fun distance(p: Point) = hypot(max(max(left-p.x,p.x-right),0.0),max(max(top-p.y,p.y-bottom),0.0))
    fun clip(w: Int,h: Int) = Box(left.coerceIn(0.0,w.toDouble()),top.coerceIn(0.0,h.toDouble()),
        right.coerceIn(0.0,w.toDouble()),bottom.coerceIn(0.0,h.toDouble()))
    fun intersection(other: Box) = max(0.0,min(right,other.right)-max(left,other.left))*
        max(0.0,min(bottom,other.bottom)-max(top,other.top))
    fun iou(other: Box): Double { val i=intersection(other); return i/max(area+other.area-i,1e-9) }
    fun contains(p: Point) = p.x in left..right && p.y in top..bottom
    companion object {
        fun around(points: List<Point>) = Box(points.minOf{it.x},points.minOf{it.y},points.maxOf{it.x},points.maxOf{it.y})
    }
}
data class Letterbox(val width: Int,val height: Int,val size: Int) {
    val scale = min(size.toDouble()/width,size.toDouble()/height)
    val resizedWidth = round(width*scale).toInt()
    val resizedHeight = round(height*scale).toInt()
    val left = (size-resizedWidth)/2
    val top = (size-resizedHeight)/2
    fun original(p: Point) = Point((p.x-left)/scale,(p.y-top)/scale)
    fun original(b: Box) = Box((b.left-left)/scale,(b.top-top)/scale,(b.right-left)/scale,(b.bottom-top)/scale).clip(width,height)
}
data class ObjectDetection(val box: Box,val label: String,val confidence: Double,
                           val coefficients: FloatArray = FloatArray(0),val networkBox: Box = box)

/** Raw YOLO11-seg output, channel-major [1,4+classes+32,candidates]. */
fun decodeYolo(raw: FloatArray, count: Int, labels: List<String>, mapping: Letterbox,
               personThreshold: Double=.5, objectThreshold: Double=.35): List<ObjectDetection> {
    require(raw.size==(4+labels.size+32)*count)
    val candidates=ArrayList<ObjectDetection>()
    for(i in 0 until count) {
        var cls=0; var score=Float.NEGATIVE_INFINITY
        for(c in labels.indices) { val p=raw[(4+c)*count+i]; if(p>score) { score=p; cls=c } }
        if(!score.isFinite() || score < if(labels[cls]=="person") personThreshold else objectThreshold) continue
        val x=raw[i].toDouble(); val y=raw[count+i].toDouble()
        val w=raw[2*count+i].toDouble(); val h=raw[3*count+i].toDouble()
        if(!x.isFinite() || !y.isFinite() || !w.isFinite() || !h.isFinite() || w<=0 || h<=0) continue
        val network=Box(x-w/2,y-h/2,x+w/2,y+h/2)
        val original=mapping.original(network)
        if(original.area<=0) continue
        val coefficients=FloatArray(32) { raw[(4+labels.size+it)*count+i] }
        candidates.add(ObjectDetection(original,labels[cls],score.toDouble(),coefficients,network))
    }
    val kept=ArrayList<ObjectDetection>()
    for(d in candidates.sortedByDescending{it.confidence}.take(300)) {
        if(kept.none { it.label==d.label && it.box.iou(d.box)>.45 }) kept.add(d)
        if(kept.size>=64) break
    }
    return kept
}

data class VisibleMask(val cells: BooleanArray,val width: Int,val height: Int,val mapping: Letterbox) {
    val count get()=cells.count{it}
    val pixelArea get()=count * (mapping.size.toDouble()/width/mapping.scale).pow(2)
    val uncertaintyPx get()=hypot(mapping.size.toDouble()/width,mapping.size.toDouble()/height)/mapping.scale/2
    fun point(i: Int) = mapping.original(Point((i%width+.5)*mapping.size/width,(i/width+.5)*mapping.size/height))
    fun distance(polygon: List<Point>): Double {
        if(polygon.isEmpty()) return Double.NaN
        var best=Double.POSITIVE_INFINITY
        for(i in cells.indices) if(cells[i]) best=min(best,pointRegionDistance(point(i),polygon))
        return if(best.isFinite()) best else Double.NaN
    }
    fun dominantFraction(): Double {
        val seen=BooleanArray(cells.size); val queue=IntArray(cells.size); var largest=0; var total=0
        for(i in cells.indices) if(cells[i] && !seen[i]) {
            var head=0; var tail=1; queue[0]=i; seen[i]=true
            while(head<tail) {
                val p=queue[head++]; val x=p%width; val y=p/width
                for(dy in -1..1) for(dx in -1..1) {
                    val nx=x+dx; val ny=y+dy
                    if(nx !in 0 until width || ny !in 0 until height) continue
                    val k=ny*width+nx
                    if(cells[k] && !seen[k]) { seen[k]=true; queue[tail++]=k }
                }
            }
            total+=tail; largest=max(largest,tail)
        }
        return if(total>0) largest.toDouble()/total else 0.0
    }
    fun iou(other: VisibleMask): Double {
        if(width!=other.width || height!=other.height || mapping!=other.mapping) return 0.0
        var intersection=0; var union=0
        for(i in cells.indices) { if(cells[i] && other.cells[i]) intersection++; if(cells[i] || other.cells[i]) union++ }
        return intersection.toDouble()/max(union,1)
    }
}

fun decodeMask(proto: FloatArray, width: Int,height: Int,d: ObjectDetection,mapping: Letterbox): VisibleMask {
    require(proto.size==32*width*height && d.coefficients.size==32)
    val cells=BooleanArray(width*height)
    val box=d.networkBox
    val x0=floor(box.left*width/mapping.size).toInt().coerceIn(0,width)
    val y0=floor(box.top*height/mapping.size).toInt().coerceIn(0,height)
    val x1=ceil(box.right*width/mapping.size).toInt().coerceIn(0,width)
    val y1=ceil(box.bottom*height/mapping.size).toInt().coerceIn(0,height)
    for(y in y0 until y1) for(x in x0 until x1) {
        val networkPoint=Point((x+.5)*mapping.size/width,(y+.5)*mapping.size/height)
        if(networkPoint.x<box.left || networkPoint.x>=box.right || networkPoint.y<box.top || networkPoint.y>=box.bottom) continue
        val p=mapping.original(networkPoint)
        if(p.x !in 0.0..mapping.width.toDouble() || p.y !in 0.0..mapping.height.toDouble()) continue
        val index=y*width+x
        var score=0f
        for(c in 0 until 32) score+=d.coefficients[c]*proto[c*width*height+index]
        cells[index]=score.isFinite() && score>0 // sigmoid(score) > .5
    }
    return VisibleMask(cells,width,height,mapping)
}

fun pointRegionDistance(p: Point, polygon: List<Point>): Double {
    if(!p.x.isFinite() || !p.y.isFinite() || polygon.isEmpty() || polygon.any{!it.x.isFinite() || !it.y.isFinite()}) return Double.NaN
    var positive=false; var negative=false; var best=Double.POSITIVE_INFINITY;var area=0.0
    for(i in polygon.indices) {
        val a=polygon[i]; val b=polygon[(i+1)%polygon.size]
        val dx=b.x-a.x; val dy=b.y-a.y
        area+=a.x*b.y-a.y*b.x
        val cross=dx*(p.y-a.y)-dy*(p.x-a.x)
        if(cross>1e-8) positive=true
        if(cross< -1e-8) negative=true
        val t=(((p.x-a.x)*dx+(p.y-a.y)*dy)/max(dx*dx+dy*dy,1e-9)).coerceIn(0.0,1.0)
        best=min(best,hypot(p.x-a.x-t*dx,p.y-a.y-t*dy))
    }
    return if(abs(area)>1e-8 && !(positive && negative)) 0.0 else best
}

fun convexHull(points: List<Point>): List<Point> {
    val sorted=points.distinct().sortedWith(compareBy<Point>{it.x}.thenBy{it.y})
    if(sorted.size<=2) return sorted
    fun cross(a:Point,b:Point,c:Point)=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)
    fun half(values:List<Point>):List<Point> {
        val out=mutableListOf<Point>()
        for(p in values){while(out.size>=2 && cross(out[out.size-2],out.last(),p)<=0)out.removeAt(out.lastIndex);out.add(p)}
        return out.dropLast(1)
    }
    return half(sorted)+half(sorted.reversed())
}

data class PersonTrack(val id: Int,val box: Box,val time: Long,val ambiguous: Boolean=false,val historyReset: Boolean=false)
class PersonTracker(private val maxPeople: Int=4,private val holdMs: Long=3000) {
    var tracks=mapOf<Int,PersonTrack>(); private set
    private var nextId=1
    fun reset() { tracks=emptyMap() }
    private fun plausible(a: PersonTrack,b: Box,now: Long): Boolean {
        val dt=(now-a.time)/1000.0
        if(dt<=0 || min(min(a.box.width,b.width),min(a.box.height,b.height))<=0) return false
        if(max(a.box.width,b.width)/min(a.box.width,b.width)>2 || max(a.box.height,b.height)/min(a.box.height,b.height)>2) return false
        val limit=min(1.0,.2+4*min(dt,.25))
        return abs(a.box.center.x-b.center.x)/min(a.box.width,b.width)<=limit &&
            abs(a.box.center.y-b.center.y)/min(a.box.height,b.height)<=limit
    }
    fun update(detections: List<ObjectDetection>,now: Long): Pair<List<PersonTrack>,Set<Int>> {
        val unique=ArrayList<ObjectDetection>()
        for(d in detections.filter{it.label=="person"}.sortedByDescending{it.box.area}) {
            if(unique.none { d.box.intersection(it.box)>=.8*d.box.area && d.box.width/it.box.width>=.75 &&
                        d.box.height/it.box.height<=.8 && abs(d.box.top-it.box.top)<=.15*it.box.height }) unique.add(d)
        }
        val overflow=unique.size>maxPeople
        val people=unique.sortedByDescending{it.confidence}.take(maxPeople)
        val old=tracks.filterValues{ now-it.time in 1..holdMs }
        val scores=people.map { d -> old.values.filter{plausible(it,d.box,now)}.map{it.box.iou(d.box) to it.id}.sortedByDescending{it.first} }
        val choices=mutableMapOf<Int,Int>()
        scores.forEachIndexed { i,s -> if(s.isNotEmpty() && s[0].first>=.25) choices[i]=s[0].second }
        val freePeople=people.indices.filter{it !in choices}; val freeTracks=old.keys.filter{it !in choices.values}
        var recovered: Int?=null
        if(freePeople.size==1 && freeTracks.size==1 && plausible(old.getValue(freeTracks[0]),people[freePeople[0]].box,now)) {
            choices[freePeople[0]]=freeTracks[0]; recovered=freePeople[0]
        }
        val retired=(tracks.keys-old.keys).toMutableSet()
        if(people.size==1 && old.size==1 && choices.isEmpty()) retired.addAll(old.keys)
        val active=people.mapIndexed { i,d ->
            val s=scores[i]
            val ambiguous=overflow || people.indices.any{j->i!=j && d.box.iou(people[j].box)>.4} ||
                (s.size>1 && s[0].first>=.25 && s[0].first-s[1].first<.15) ||
                (i in choices && choices.values.count{it==choices[i]}>1)
            if(ambiguous) retired.addAll(s.filter{it.first>=.25}.map{it.second})
            val id=if(ambiguous) nextId++ else choices[i] ?: nextId++
            PersonTrack(id,d.box,now,ambiguous,i==recovered)
        }
        retired.removeAll(active.map{it.id}.toSet())
        tracks=old.filterKeys{it !in retired && it !in active.map{t->t.id}}+active.associateBy{it.id}
        return active to retired
    }
}
