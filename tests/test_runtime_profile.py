import pandas as pd
from childsafe.evaluation.runtime_profile import runtime_profile


def test_people_do_not_inflate_rate_or_stage_timings():
    rows = [dict(session_id=1,person_id=p,t=t,detector_s=.02) for p in (1,2,3,4) for t in (0,.1,.2,.6)]
    report = runtime_profile(pd.DataFrame(rows))
    assert report['samples'] == 4 and report['processed_fps'] == 5
    assert report['continuity_breaks'] == 1 and report['detector_s']['median'] == .02
