"""User-facing scope, distinct from dataset labels and medical diagnosis."""

_NAMES = {'knife':'dao', 'fork':'dĩa', 'scissors':'kéo'}


def product_scope(cfg):
    categories = cfg.get('hazard_policy', {}).get('mouth_hazards', {})
    return {
        'purpose':'Hỗ trợ người chăm sóc chú ý khi vật cầm tay tiến sát vùng miệng trên ảnh.',
        'high_alert_categories':list(categories),
        'high_alert_names':[_NAMES.get(name, name) for name in categories],
        'unlocalized_examples':['bút', 'đũa', 'đồng xu', 'pin cúc'],
        'observation_limit':'Quay lưng hoặc che miệng có thể làm bỏ sót; không đủ thông tin không có nghĩa an toàn.',
        'validation':'Nguyên mẫu; hiệu quả cảnh báo trên trẻ và camera sử dụng thực tế chưa được kiểm chứng.',
        'measurement':'Khoảng cách là tỷ lệ 2D, không xác nhận tiếp xúc, nuốt hoặc nghẹn.',
    }


def caregiver_action(level, observation='unobservable', green=False):
    """Never let limited observation erase the action for an existing high alert."""
    if level >= 2:
        return 'Kiểm tra trực tiếp người được theo dõi và vật đang cầm; không chờ ứng dụng xác nhận tiếp xúc.'
    if level == 1:
        return 'Chú ý và kiểm tra tình huống đang diễn ra; đây là tín hiệu nghi ngờ, chưa xác nhận tiếp xúc.'
    if not green or observation != 'observable':
        return 'Kiểm tra trực tiếp khi cần; điều chỉnh góc nhìn hoặc ánh sáng nếu vùng miệng bị che hay quá nhỏ.'
    return 'Tiếp tục giám sát; chưa ghi nhận bằng chứng nguy cơ trong phạm vi hệ thống quan sát được.'
