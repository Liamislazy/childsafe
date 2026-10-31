package vn.childsafe.core

/** Evidence is about visible image regions, never physical distance or choking. */
data class MouthEvidence(
    val mouthObservable: Boolean,
    val imageClear: Boolean,
    val objectLocalized: Boolean,
    val handAssociated: Boolean,
    val nearCandidate: Boolean,
    val nearReliableMask: Boolean,
    val maskStable: Boolean,
    val knownHazard: Boolean,
    val categoryConflict: Boolean = false,
    val ambiguousPerson: Boolean = false,
    val observedDeparture: Boolean = false,
    val knownBenignObject: Boolean = false,
)

data class CameraDecision(val risk: Risk, val observation: Observation,
                          val reason: String, val notify: Boolean, val episode: Long)

/** One gate per temporary person ID; resets on identity ambiguity/reconnect. */
class CameraRiskGate(
    private val maxGapMs: Long = 250,
    private val holdMs: Long = 800,
    private val cautionMs: Long = 150,
    private val dangerMs: Long = 500,
    private val cooldownMs: Long = 5000,
    private val calibrated: Boolean = false,
    private val benignDwellMs: Long = 3000,
) {
    private var previous: Long? = null
    private var candidateSince: Long? = null
    private var reliableSince: Long? = null
    private var lastEvidence: Long? = null
    private var lastStrongEvidence: Long? = null
    private var risk = Risk.NORMAL
    private var lastNotification: Long? = null
    private var episode = 0L

    fun update(e: MouthEvidence, now: Long): CameraDecision {
        require(previous == null || now >= previous!!)
        val continuous = previous == null || now - previous!! <= maxGapMs
        previous = now
        if (!continuous || e.ambiguousPerson) {
            candidateSince = null
            reliableSince = null
        }
        if (!continuous && !e.ambiguousPerson)
            return lost(now, "Khoảng giữa hai ảnh vượt giới hạn; không dùng ảnh mới để nối lịch sử cũ")
        if (e.ambiguousPerson) {
            // Never carry another person's history through an ambiguous association.
            risk = Risk.NORMAL
            lastEvidence = null
            lastStrongEvidence = null
            episode++
            return CameraDecision(risk, Observation.UNOBSERVABLE, "Không gán chắc được tay/vật cho từng người", false, episode)
        }
        if (!e.mouthObservable || !e.imageClear) {
            candidateSince = null
            reliableSince = null
            return lost(now, "Chưa quan sát rõ vùng miệng hoặc chất lượng ảnh chưa đủ")
        }
        if (e.observedDeparture) {
            candidateSince = null
            reliableSince = null
            lastEvidence = null
            lastStrongEvidence = null
            if (risk != Risk.NORMAL) episode++
            risk = Risk.NORMAL
            return decision("Vật đã rời vùng miệng trên ảnh", false)
        }
        // A hand at the face with no localized object cannot renew an event.
        val candidate = e.objectLocalized && e.handAssociated && e.nearCandidate
        if (!candidate) {
            candidateSince = null
            reliableSince = null
            return lost(now, "Chưa có bằng chứng vật cầm tay sát vùng miệng")
        }
        if (candidateSince == null) candidateSince = now
        lastEvidence = now
        val strong = e.nearReliableMask && e.maskStable && e.knownHazard && !e.categoryConflict
        if (strong) lastStrongEvidence = now
        if (!strong) reliableSince = null
        else if (reliableSince == null) reliableSince = now
        val proposed = when {
            reliableSince != null && now - reliableSince!! >= dangerMs -> Risk.DANGER
            now - candidateSince!! >= (if(e.knownBenignObject) benignDwellMs else cautionMs) -> Risk.CAUTION
            else -> Risk.NORMAL
        }
        val old = risk
        val highEvidenceLost = risk == Risk.DANGER && !strong &&
            (lastStrongEvidence == null || now - lastStrongEvidence!! > holdMs)
        if (highEvidenceLost) { risk = Risk.CAUTION; episode++ }
        // Weaker current classification does not prove a known danger has ended.
        if (proposed.ordinal > risk.ordinal) risk = proposed
        if (old == Risk.NORMAL && risk != Risk.NORMAL) episode++
        val notify = risk.ordinal > old.ordinal && (lastNotification == null ||
            now - lastNotification!! >= cooldownMs || risk == Risk.DANGER)
        if (notify) lastNotification = now
        val reason = when {
            highEvidenceLost -> "Mất bằng chứng mask/nguy cơ cao; chỉ còn bằng chứng nghi ngờ, cần kiểm tra trực tiếp"
            risk == Risk.DANGER -> "Vật thuộc nhóm nguy hiểm có mask ổn định sát vùng miệng trên ảnh"
            e.categoryConflict -> "Hai mô hình chưa thống nhất loại vật; cần kiểm tra trực tiếp"
            risk == Risk.CAUTION -> "Có vật nghi ngờ sát vùng miệng; chưa đủ bằng chứng nguy cơ cao"
            else -> "Đang tích lũy bằng chứng vị trí vật"
        }
        return decision(reason, notify)
    }

    private fun lost(now: Long, reason: String): CameraDecision {
        val age = (if (risk == Risk.DANGER) lastStrongEvidence else lastEvidence)?.let { now - it }
        if (age != null && age <= holdMs && risk != Risk.NORMAL)
            return CameraDecision(risk, Observation.DEGRADED, "$reason; giữ bằng chứng gần nhất trong thời gian ngắn", false, episode)
        if (risk != Risk.NORMAL) episode++
        risk = Risk.NORMAL
        lastEvidence = null
        lastStrongEvidence = null
        return CameraDecision(risk, Observation.UNOBSERVABLE, "$reason; không kết luận an toàn", false, episode)
    }

    private fun decision(reason: String, notify: Boolean) = CameraDecision(risk,
        if (calibrated) Observation.OBSERVABLE else Observation.DEGRADED,
        if (calibrated) reason else "$reason. Ngưỡng Android chưa được kiểm chứng", notify, episode)
}
