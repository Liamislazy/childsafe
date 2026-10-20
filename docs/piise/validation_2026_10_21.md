# Kiểm chứng phần bổ sung ngày 21 tháng 10 năm 2026

## Đã kiểm tra

- Toàn bộ suite: **214 passed, 3 skipped**, 17 cảnh báo dependency, 18,57 giây.
  Có hồi quy cho checksum model GPU được chọn, detector truyền trực tiếp, manifest
  và bản web; CUDA được mô phỏng, không có GPU thật trên máy này.
- Các bảng minh chứng dùng evaluator nghiêm ngặt hiện có. Ba test riêng kiểm tra
  không có phiên không tạo số 0 giả; phiên chưa hoàn tất bị từ chối; phiên chỉ có
  âm tính giữ recall là chưa xác định và tách ba chính sách cảnh báo.
- Kiểm tra model local ghi đủ hash: classifier robust + other, YOLO11n-seg,
  MediaPipe mặt/tay. Không tải model mới hoặc mở camera.
- Model thật suy luận trên ảnh hình học không người: 2 warmup, 8 mẫu, CPU 2
  threads, ảnh 640×480. Median/p90 (ms): YOLO 82,25/85,01; classifier 27,36/28,81;
  MediaPipe 20,78/21,22; pipeline cảnh không người 82,08/85,09.
  Artifact local: `outputs/piise_evidence/runtime_smoke.json`. Đây không phải FPS
  camera nhiều người; pipeline cảnh trống không chạy đầy đủ workload mốc/tay/vật.
  Không dùng số này để tuyên bố tăng tốc so với lần đo cũ có tải máy khác.
- Edge headless: kiểm tra metadata model thật và mục phạm vi, không lỗi JavaScript;
  hồi quy phiên cũ renew chậm không ghi đè/dừng phiên mới thành công. Video trong
  kiểm tra là canvas giả, không camera hoặc micro thật. Máy chủ thử đã dừng.
- Chạy CLI xuất minh chứng trên thư mục chưa có phiên: trạng thái `not_measured`,
  CSV chỉ có tiêu đề, recall/FA/latency camera chưa đo.

## Hành vi thay đổi

Checksum detector trong web/manifest khớp model được chọn thay vì luôn lấy model
CPU. API và giao diện nêu tên model thực tế, fallback, phạm vi nhóm vật, cùng việc
người chăm sóc nên kiểm tra ở từng mức. Không đổi checkpoint, ngưỡng nguy cơ,
quy tắc quan sát, số người tối đa hoặc cách xử lý mất dấu.

Có công cụ kiểm tra tệp model và xuất bảng dễ đọc từ các phiên có nhãn/provenance.
Các tài liệu chuẩn bị riêng gồm phạm vi, demo, poster, phản biện và việc còn lại.
Báo cáo đã chốt trên Drive không được chỉnh sửa.

## Còn chưa kiểm chứng

Hiệu quả camera thật sau sửa; góc cao, quay lưng, hai/bốn người; tốc độ và độ trễ
toàn luồng khi có người; recall sự kiện; FA/giờ; hiệu quả với trẻ; GPU thật; detector
bút/đồng xu/pin cúc; phản hồi người chăm sóc và tác động xã hội. Không bổ sung dữ
liệu, không train mới, không ghi hình người trong đợt này.
