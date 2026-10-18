"""Presentation labels; numerical evidence and replay reasons remain unchanged."""
REASONS = {
    'object mask not temporally stable':'Vùng vật chưa ổn định qua thời gian',
    'object near mouth; category does not establish safety':'Vật gần vùng miệng; tên loại vật không xác nhận an toàn',
    'no observable person':'Chưa có người quan sát được',
    'no observable person; detector limited':'Chưa có người quan sát được; bộ định vị đang hạn chế',
    'poor exposure':'Ánh sáng vùng mặt không phù hợp',
    'blurred image':'Vùng mặt bị nhòe',
    'face too small':'Khuôn mặt có quá ít pixel ảnh gốc',
    'mouth too small':'Vùng miệng có quá ít pixel ảnh gốc',
    'hand too small':'Bàn tay có quá ít pixel ảnh gốc',
    'no observable hand':'Chưa quan sát rõ bàn tay',
    'head orientation unavailable':'Chưa xác định được hướng mặt',
    'mouth hidden by head orientation':'Hướng mặt không cho phép quan sát miệng',
    'unstable mouth landmarks':'Vị trí miệng chưa ổn định',
    'mouth not observable':'Chưa quan sát rõ miệng',
    'ambiguous association':'Chưa xác định chắc tay hoặc vật thuộc người nào',
    'detector unavailable':'Bộ định vị người/vật chưa sẵn sàng',
    'classifier unavailable':'Bộ phân loại vật chưa sẵn sàng',
    'elevated-view thresholds unvalidated':'Ngưỡng camera góc cao chưa được kiểm chứng',
    'object class coverage limited':'Các lớp vật được hỗ trợ còn hạn chế',
    'no reliable held-object mask evidence':'Chưa có vùng vật đủ tin cậy để xác nhận vị trí',
    'stale or invalid frame':'Ảnh đã cũ hoặc không hợp lệ',
    'source disconnected':'Nguồn camera bị ngắt',
    'frame expired during inference':'Ảnh đã hết hạn trong lúc xử lý',
    'object crop too small for classification':'Vùng vật quá nhỏ để phân loại',
    'landmark model unavailable':'Bộ xác định mốc chưa sẵn sàng',
}


def reason_vi(reason):
    return REASONS.get(reason,reason)
