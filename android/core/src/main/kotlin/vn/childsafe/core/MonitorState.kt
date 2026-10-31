package vn.childsafe.core

enum class Risk { NORMAL, CAUTION, DANGER }
enum class Observation { OBSERVABLE, DEGRADED, UNOBSERVABLE }
enum class Connection { FRESH, STALE, OFFLINE }

data class CameraResult(
    val epoch: String,
    val sequence: Long,
    val eventId: String,
    val risk: Risk,
    val observation: Observation,
    val reason: String,
    val isAlert: Boolean,
)

data class MonitorView(
    val connection: Connection,
    val risk: Risk?,
    val observation: Observation,
    val reason: String,
    val notify: Boolean = false,
    val validUntilElapsedMs: Long = Long.MIN_VALUE,
) {
    val green: Boolean get() = connection == Connection.FRESH && risk == Risk.NORMAL &&
        observation == Observation.OBSERVABLE
    fun atTime(now: Long): MonitorView = if (connection == Connection.FRESH && now >= validUntilElapsedMs)
        copy(connection = Connection.STALE, observation = Observation.UNOBSERVABLE, notify = false) else this
}

/** One instance per camera. All times are elapsed-real-time milliseconds. */
class MonitorState(private val freshMs: Long = 2500, private val cooldownMs: Long = 5000) {
    private var epoch: String? = null
    private var sequence = -1L
    private var result: CameraResult? = null
    private var freshUntil = Long.MIN_VALUE
    private var online = false
    private var lastClock = Long.MIN_VALUE
    private var mutedUntil = Long.MIN_VALUE
    private var lastNotification = Long.MIN_VALUE
    private var lastNotifiedRisk = Risk.NORMAL
    private val seenEvents = LinkedHashSet<String>()
    private val seenTests = LinkedHashSet<String>()

    fun beginEpoch(value: String) {
        if (epoch == value) return
        epoch = value
        sequence = -1
        result = null
        online = false
        freshUntil = Long.MIN_VALUE
        // Keep acknowledged event IDs through reconnect to prevent replay sounds.
    }

    fun muteFor(now: Long, durationMs: Long) {
        require(durationMs in 0..3_600_000)
        mutedUntil = now + durationMs
    }

    fun disconnected(now: Long): MonitorView {
        checkClock(now)
        online = false
        return view(now)
    }

    fun accept(value: CameraResult, expectedEpoch: String, ageMs: Long, now: Long,
               bootstrap: Boolean = false): MonitorView {
        checkClock(now)
        beginEpoch(expectedEpoch)
        if (value.epoch != expectedEpoch || value.sequence <= sequence || ageMs !in 0 until freshMs) {
            return view(now)
        }
        sequence = value.sequence
        result = value
        online = true
        freshUntil = now + freshMs - ageMs
        val newEvent = seenEvents.add(value.eventId)
        while (seenEvents.size > 256) seenEvents.remove(seenEvents.first())
        val cooldownPassed = lastNotification == Long.MIN_VALUE || now - lastNotification >= cooldownMs
        val escalated = value.risk.ordinal > lastNotifiedRisk.ordinal
        val notify = !bootstrap && value.isAlert && newEvent && value.risk != Risk.NORMAL &&
            now >= mutedUntil && (cooldownPassed || escalated || value.risk == Risk.DANGER)
        if (notify) { lastNotification = now; lastNotifiedRisk = value.risk }
        return view(now).copy(notify = notify)
    }

    fun view(now: Long): MonitorView {
        checkClock(now)
        val connection = if (!online) Connection.OFFLINE else if (now >= freshUntil) Connection.STALE else Connection.FRESH
        return MonitorView(connection, result?.risk,
            if (connection == Connection.FRESH) result!!.observation else Observation.UNOBSERVABLE,
            result?.reason.orEmpty(), validUntilElapsedMs = freshUntil)
    }

    fun acceptTest(eventId: String, messageEpoch: String, ageMs: Long, now: Long,
                   bootstrap: Boolean = false): Boolean {
        checkClock(now)
        if(bootstrap || messageEpoch!=epoch || ageMs !in 0 until freshMs || now<mutedUntil) return false
        val fresh=seenTests.add(eventId)
        while(seenTests.size>64) seenTests.remove(seenTests.first())
        return fresh // No risk, observation, sequence or connection mutation.
    }

    private fun checkClock(now: Long) {
        require(now >= lastClock) { "Use monotonic elapsed time" }
        lastClock = now
    }
}
