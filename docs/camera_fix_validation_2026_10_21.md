# Kiểm tra lượt sửa camera — 21/10/2026

## Kiểm thử đã chạy

- Bộ Python sau sửa: **252 passed, 3 skipped**, 17 cảnh báo deprecation từ dependency,
  24,06 giây. Ba kiểm thử tích hợp theo opt-in không chạy; không diễn giải chúng là đạt.
- Sau chỉnh nhỏ parser CSV/nhãn trống: chạy lại 12 kiểm thử workflow dữ liệu/notebook, đều đạt.
- Edge headless trên server local tạm 127.0.0.1:8767: kiểm tra danh mục lớp thực tế,
  nền xám khi quan sát hạn chế, nguồn detector độc lập, dùng frame mới đã có sẵn; đạt.
- Edge với luồng canvas giả lập: bật/tắt và renewal cũ không ghi đè phiên mới; đạt,
  không có lỗi JavaScript. Không gọi webcam vật lý. Server thử đã dừng sau kiểm tra.

Các regression mới kiểm tra 15 tên lớp bằng bằng chứng **tổng hợp** để xác minh luật,
không xác minh model có nhận được 15 loại vật. Kiểm tra thêm detector hộp không tạo
mask/dwell xác nhận, model nhận khác loại chỉ CAUTION, hash cả hai checkpoint và replay.

## Đo xử lý trên ảnh người dùng cung cấp

Ảnh đầu vào lấy từ phần ảnh camera của screenshot người dùng, chỉ crop trong RAM,
1122×637. Đây là ảnh tĩnh lặp, không có timestamp sensor hay sự kiện mới. Dùng checkpoint
classifier robust+other và YOLO11n-seg local; CUDA không khả dụng. Mỗi lượt so sánh chính
warm-up 2 lần, đo 30 lần, thời gian t đầu vào mô phỏng 0,2 giây/bước.

| Cấu hình cache classifier | Median xử lý | p90 xử lý |
|---|---:|---:|
| 0,1 giây (cấu hình cũ), 8 luồng thực tế | 233,47 ms | 249,62 ms |
| 0,3 giây (cấu hình mới), 8 luồng thực tế | 227,97 ms | 254,22 ms |

So sánh này giữ runtime ở commit `6d59838` và thay riêng khoảng cache; chưa có thử nghiệm
lặp nhiều phiên/kiểm soát tải nền. Median giảm nhỏ, p90 tăng; **chưa chứng minh cải thiện
tốc độ tổng thể**. Không quy đổi các số này thành FPS camera hoặc độ trễ cảnh báo.
Các số không tính toàn bộ JPEG/browser/vòng gửi nhận. Ngưỡng tuổi frame/khoảng gián đoạn
vẫn 250 ms nên CPU vẫn có thể bỏ frame và không đủ liên tục để tạo DANGER.

Thử ban đầu 8 mẫu với 2/4/8 luồng và ảnh thu nhỏ cho MediaPipe không cho thấy lợi ích rõ.
Không giữ ép 2 luồng hay giảm ảnh đầu vào bộ mốc làm mặc định. Giữ ảnh gốc; chỉ cache
phân loại cùng vật trong thời gian ngắn và bỏ chờ frame/trình vẽ dư thừa trên web.
Raw số liệu đo cục bộ nằm trong `outputs/camera_runtime_fix/`, không commit ảnh hoặc trọng số.

YOLO vẫn nhận sai bút trong ảnh thử thành `toothbrush`, điểm khoảng 0,425. Ảnh này
chưa chứng minh cải thiện nhận bút, kéo ở xa hay recall sự kiện sau sửa.

## Chưa thực hiện

- Chưa thu phiên camera thật sau sửa, chưa đo recall sự kiện/FA mỗi giờ/coverage trên người.
- Chưa có dữ liệu segmentation đã review cho 12 lớp mới; chưa train hoặc chạy notebook Kaggle.
- Chưa hiệu chuẩn các ngưỡng camera hoặc detector 0,75; không đổi cờ calibrated/coverage thành true.
- Chưa xác nhận vị trí đặt camera 2 m là phù hợp. Cần thử theo pixel vật/mặt và điều kiện thực tế.

Không tải dataset lớn, không bật micro, không ghi video, không thay báo cáo Drive đã chốt.
Đọc `docs/expanded_hazard_runtime.md` để chuẩn bị nhãn và train model mới.
