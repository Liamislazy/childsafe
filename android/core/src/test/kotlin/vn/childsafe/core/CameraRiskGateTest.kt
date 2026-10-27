package vn.childsafe.core

import org.junit.Assert.*
import org.junit.Test

class CameraRiskGateTest {
    private val strong = MouthEvidence(true,true,true,true,true,true,true,true)
    @Test fun handAtFaceWithoutObjectNeverCreatesAlert() {
        val gate = CameraRiskGate()
        for(t in 0L..3000L step 100) {
            val result = gate.update(strong.copy(objectLocalized = false),t)
            assertEquals(Risk.NORMAL,result.risk)
            assertFalse(result.notify)
        }
    }
    @Test fun boxesAndConflictingClassesCannotCreateHighRisk() {
        for(weak in listOf(strong.copy(nearReliableMask=false),strong.copy(categoryConflict=true),strong.copy(maskStable=false))) {
            val gate = CameraRiskGate()
            var result = gate.update(weak,0)
            for(t in 100L..3000L step 100) result=gate.update(weak,t)
            assertEquals(Risk.CAUTION,result.risk)
        }
    }
    @Test fun noMouthNoNewContactAndOldEvidenceExpires() {
        val gate = CameraRiskGate()
        for(t in 0L..600L step 100) gate.update(strong,t)
        assertEquals(Risk.DANGER,gate.update(strong.copy(mouthObservable=false),700).risk)
        val ended=gate.update(strong.copy(mouthObservable=false),1500)
        assertEquals(Risk.NORMAL,ended.risk)
        assertEquals(Observation.UNOBSERVABLE,ended.observation)
        assertEquals(Risk.NORMAL,CameraRiskGate().update(strong.copy(mouthObservable=false),0).risk)
    }
    @Test fun temporalGapBreaksDwellAndAmbiguityDoesNotTransferHistory() {
        val gate = CameraRiskGate()
        gate.update(strong,0); gate.update(strong,200)
        assertNotEquals(Risk.DANGER,gate.update(strong,900).risk)
        assertEquals(Risk.NORMAL,gate.update(strong.copy(ambiguousPerson=true),1000).risk)
        assertNotEquals(Risk.DANGER,gate.update(strong,1100).risk)
    }
    @Test fun timeThresholdsAtDifferentFrameRates() {
        for(fps in listOf(5,10,15,30)) {
            val gate=CameraRiskGate()
            var transition: Long?=null
            for(frame in 0..fps) {
                val t=frame*1000L/fps
                if(gate.update(strong,t).risk==Risk.DANGER && transition==null) transition=t
            }
            assertNotNull(transition)
            assertTrue(transition!! in 500L..(500L+1000L/fps+1))
        }
    }
    @Test fun departureEndsRiskCooldownDoesNotSuppressRisk() {
        val gate=CameraRiskGate()
        for(t in 0L..600L step 100) gate.update(strong,t)
        val weak=gate.update(strong.copy(knownHazard=false),700)
        assertEquals(Risk.DANGER,weak.risk)
        assertFalse(weak.notify)
        assertEquals(Risk.NORMAL,gate.update(strong.copy(observedDeparture=true),800).risk)
    }
    @Test fun weakCandidateCannotKeepHighRiskEvidenceAliveForever() {
        val gate=CameraRiskGate()
        for(t in 0L..600L step 100) gate.update(strong,t)
        var result=gate.update(strong.copy(nearReliableMask=false),700)
        for(t in 800L..1600L step 100) result=gate.update(strong.copy(nearReliableMask=false),t)
        assertEquals(Risk.CAUTION,result.risk)
    }
    @Test fun slowFramesCannotRenewAnOldDanger() {
        val gate=CameraRiskGate()
        for(t in 0L..600L step 100) gate.update(strong,t)
        var result=gate.update(strong,1000)
        for(t in 1400L..2600L step 400) result=gate.update(strong,t)
        assertEquals(Risk.NORMAL,result.risk)
        assertEquals(Observation.UNOBSERVABLE,result.observation)
    }
    @Test fun recognizedBenignObjectDoesNotAlertImmediatelyDuringNormalEating() {
        val gate=CameraRiskGate()
        val safe=strong.copy(knownHazard=false,knownBenignObject=true)
        for(t in 0L..2900L step 100) assertEquals(Risk.NORMAL,gate.update(safe,t).risk)
        assertEquals(Risk.CAUTION,gate.update(safe,3000).risk)
    }
}
