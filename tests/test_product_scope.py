from childsafe.config import load_config
from childsafe.product import caregiver_action, product_scope


def test_high_alert_action_remains_visible_when_observation_is_lost():
    message = caregiver_action(2, 'unobservable', False)
    assert 'Kiểm tra trực tiếp' in message
    assert 'điều chỉnh góc' not in message.lower()
    assert 'tiếp xúc' in message


def test_low_risk_with_missing_observation_does_not_claim_safety():
    assert 'Kiểm tra trực tiếp' in caregiver_action(0, 'unobservable', False)
    assert 'trong phạm vi' in caregiver_action(0, 'observable', True)
    cfg = load_config()
    scope = product_scope(cfg)
    assert scope['high_alert_categories'] == ['knife', 'fork', 'scissors']
    assert 'pin cúc' in scope['unlocalized_examples']
