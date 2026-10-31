package vn.childsafe.core
import org.junit.Assert.*
import org.junit.Test

class VisionGeometryTest {
    @Test fun letterboxAndCropReturnOriginalPixels() {
        val m=Letterbox(1280,720,320)
        assertEquals(70,m.top)
        assertEquals(Point(640.0,360.0),m.original(Point(160.0,160.0)))
        assertEquals(Box(0.0,0.0,1280.0,720.0),m.original(Box(0.0,70.0,320.0,250.0)))
    }
    @Test fun longMaskCanBeNearMouthEvenWhenHandIsFar() {
        val m=Letterbox(80,80,80)
        val cells=BooleanArray(80*80)
        for(x in 10..60) cells[40*80+x]=true
        val mask=VisibleMask(cells,80,80,m)
        val mouth=listOf(Point(58.0,38.0),Point(62.0,38.0),Point(62.0,42.0),Point(58.0,42.0))
        assertEquals(0.0,mask.distance(mouth),0.0)
        assertTrue(Point(10.0,40.0).distance(Point(60.0,40.0))>40)
        assertTrue(VisibleMask(BooleanArray(6400),80,80,m).distance(mouth).isNaN())
    }
    @Test fun crossingPeopleDiscardIdentityHistory() {
        val tracker=PersonTracker()
        fun person(b: Box)=ObjectDetection(b,"person",.9)
        val a=tracker.update(listOf(person(Box(0.0,0.0,100.0,200.0)),person(Box(150.0,0.0,250.0,200.0))),0).first
        val b=tracker.update(listOf(person(Box(65.0,0.0,165.0,200.0)),person(Box(85.0,0.0,185.0,200.0))),100).first
        assertTrue(b.all{it.ambiguous})
        assertTrue(b.none{t->a.any{it.id==t.id}})
    }
    @Test fun abruptReplacementDoesNotReuseAnIdentity() {
        val tracker=PersonTracker()
        val a=tracker.update(listOf(ObjectDetection(Box(0.0,0.0,100.0,200.0),"person",.9)),0).first[0]
        val b=tracker.update(listOf(ObjectDetection(Box(500.0,0.0,600.0,200.0),"person",.9)),20)
        assertNotEquals(a.id,b.first[0].id)
        assertTrue(a.id in b.second)
    }
    @Test fun emptyMaskAndDisconnectedIslandsCannotBeReliable() {
        val m=Letterbox(10,10,10)
        val cells=BooleanArray(100); cells[0]=true; cells[99]=true
        assertEquals(.5,VisibleMask(cells,10,10,m).dominantFraction(),0.0)
    }
    @Test fun decodeChannelMajorYoloAndPrototypeMaskAtOriginalCoordinates() {
        val count=2
        val raw=FloatArray((4+2+32)*count)
        raw[0]=160f;raw[count]=160f;raw[2*count]=80f;raw[3*count]=80f
        raw[(4+1)*count]=.9f;raw[(4+2)*count]=1f
        val mapping=Letterbox(1280,720,320)
        val detection=decodeYolo(raw,count,listOf("person","knife"),mapping).single()
        assertEquals("knife",detection.label)
        assertEquals(Box(480.0,200.0,800.0,520.0),detection.box)
        val proto=FloatArray(32*8*8)
        for(i in 0 until 64) proto[i]=1f
        val mask=decodeMask(proto,8,8,detection,mapping)
        assertEquals(4,mask.count)
        assertEquals(1.0,mask.dominantFraction(),0.0)
        assertEquals(Point(560.0,280.0),mask.point(3*8+3))
    }
    @Test fun closedMouthLineDoesNotMakeEveryCollinearObjectLookNear() {
        val mouth=listOf(Point(20.0,20.0),Point(30.0,20.0),Point(40.0,20.0),Point(30.0,20.0))
        assertEquals(15.0,pointRegionDistance(Point(5.0,20.0),mouth),0.0)
        assertEquals(0.0,pointRegionDistance(Point(30.0,20.0),mouth),0.0)
        assertEquals(2,convexHull(mouth).size)
    }
}
