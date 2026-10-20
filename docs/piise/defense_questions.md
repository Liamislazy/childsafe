# Câu hỏi nhóm cần tự giải thích

Tài liệu luyện trao đổi; mỗi thành viên tự trả lời bằng hiểu biết và nhật ký thật,
không học thuộc câu trả lời do AI tạo. Xem mã và kết quả để kiểm tra lại.

| Câu hỏi | Minh chứng hoặc phần mã cần nắm |
|---|---|
| Nhóm phát hiện điều gì và không phát hiện điều gì? | product_scope.md; `hazard_policy` trong config; phân biệt phân loại, định vị và tiếp xúc |
| Vì sao chia ngẫu nhiên gần 99% nhưng chia theo ảnh nguồn thấp hơn? | docs/split_report.md, docs/results_moduleA.md; bản sao tăng cường và nhóm nguồn |
| Vì sao chọn EfficientNet khi mô hình khác có accuracy cao hơn? | Phép đo thật về số tham số, độ bền, thời gian; ghi đúng máy, batch và số luồng |
| Nhãn safe trong dữ liệu có xác nhận an toàn cho trẻ không? | alerts/hazards.py; nguy cơ theo tình huống khác nhãn huấn luyện |
| Tay còn xa nhưng đầu vật dài gần miệng thì sao? | vision/elevated.py và objects.py; mask vật, crop về tọa độ gốc |
| Quay lưng, mất miệng hoặc không thấy miệng từ đầu thì sao? | observation.py, processor.py; sự kiện kết thúc do mất quan sát, không suy ra tiếp xúc mới |
| Hai người đi giao nhau có chuyển cảnh báo sang người khác không? | tracking.py; gán mơ hồ, reset lịch sử; giới hạn đổi người cùng vị trí |
| Camera chậm ảnh hưởng gì? | max_gap_s, stale_frame_s, runtime/browser.py; không nới ngưỡng để che máy chậm |
| Một dòng trong CSV chứng minh điều gì? | Frame/người, thời gian, config/model/source hash; không đồng nghĩa một frame độc lập |
| Recall ảnh khác recall sự kiện thế nào? | evaluation/product_report.py; sự kiện là khoảng nhãn độc lập |
| Vì sao 0 báo nhầm trong một phút chưa chứng minh ít báo nhầm? | Thời lượng âm tính ngắn, khoảng tin cậy và cảnh chưa bao phủ |
| Đối chứng khoảng cách có cùng recall với hệ thống đầy đủ không? | Chọn ngưỡng trên dev; bảng so sánh thô không tự bảo đảm cùng recall |
| Chọn model GPU nào và làm sao biết dữ liệu đo đúng model? | Detector hiện trong /api/info; cùng checksum ở CSV, session.json và manifest |
| Những kết quả nào chưa có? | remaining_work.md; không thay bằng số kiểm thử phần mềm |
| Hai thành viên đã trực tiếp làm gì và AI hỗ trợ gì? | Mục 8/10 do nhóm xác nhận, nhật ký thật, lịch sử mã và prompt log |

Một bài luyện hữu ích: lấy một cảnh báo sai, lần theo detector → mốc → vật được
gán → phân loại → điều kiện luật; chỉ ra nơi bằng chứng mất hoặc sai, không đổ
mọi lỗi cho dataset. Giữ các kết luận gắn đúng với thí nghiệm đã làm.
