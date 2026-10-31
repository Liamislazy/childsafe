"""User-facing scope, distinct from dataset labels and medical diagnosis."""

_NAMES = {'knife':'dao', 'fork':'dĩa', 'scissors':'kéo', 'pen':'bút', 'pencil':'bút chì',
          'chopsticks':'đũa', 'coin':'đồng xu', 'button_battery':'pin cúc',
          'cylindrical_battery':'pin trụ', 'screw':'ốc vít', 'nail':'đinh',
          'safety_pin':'kim băng', 'needle':'kim', 'marble':'viên bi', 'small_toy_piece':'chi tiết đồ chơi nhỏ'}


def product_scope(cfg, detector=None):
    categories = cfg.get('hazard_policy', {}).get('mouth_hazards', {})
    additional = cfg.get('hazard_policy', {}).get('supplemental_hazards', {})
    names = set(getattr(detector, 'names', [])) if detector is not None else set()
    active = [name for name in additional if name in names] if detector is not None and detector.ok else []
    pending = [name for name in additional if name not in active]
    return {
        'purpose':'Hỗ trợ người chăm sóc chú ý khi vật cầm tay tiến sát vùng miệng trên ảnh.',
        'high_alert_categories':list(categories),
        'high_alert_names':[_NAMES.get(name, name) for name in categories],
        'additional_detector_categories':active,
        'additional_detector_names':[_NAMES.get(name, name) for name in active],
        'pending_training_categories':pending,
        'pending_training_names':[_NAMES.get(name, name) for name in pending],
        'unlocalized_examples':[_NAMES[name] for name in ('pen','chopsticks','coin','button_battery') if name not in active],
        'detector_classes':sorted(names),
        'class_limit':'Có tên lớp chưa chứng minh nhận được vật nhỏ/xa; model hộp chỉ tạo cảnh giác.',
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
