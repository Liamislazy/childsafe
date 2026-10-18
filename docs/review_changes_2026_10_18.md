# Cải thiện sau rà soát mã nguồn — 18/10/2026

## 1. Danh tính và lịch sử

Tracker kiểm tra tốc độ dịch chuyển tương đối theo kích thước hộp người và thay đổi
tỷ lệ trước khi ghép ID. Phục hồi IoU thấp giữ số hiển thị nhưng reset lịch sử nguy
cơ, hình học, chất lượng và phân loại; người thay thế ở vị trí khác nhận ID mới.
Các ngưỡng là giá trị khởi đầu chưa hiệu chuẩn. Người khác xuất hiện đúng cùng vị
trí vẫn không phân biệt chắc chắn bằng tracker hình học; chưa có nhận dạng danh tính.

Kiểm chứng: các test tracking/identity/pipeline và hồi quy đổi người, đổi kích thước,
timestamp không tăng, phục hồi IoU thấp. Không có thử camera người thật ở bước này.

## 2. Đóng phiên camera

Hủy phiên và công bố kết quả dùng chung khóa ngắn. Xử lý model đang chạy được phép
kết thúc nhưng không ghi CSV hoặc giữ lịch sử nguy cơ sau hủy; thông tin phiên giữ
nguyên epoch đã xuất. Bắt đầu phiên giữ khóa xử lý trong suốt reset. Frontend ràng
buộc các tác vụ bất đồng bộ với thế hệ phiên để phản hồi cũ không tắt phiên mới.

Kiểm chứng: hồi quy dừng trong lúc giải mã và inference, quyền sở hữu/expiry,
reset khi bận, web runtime và nhãn quan sát. Chưa thử ngắt camera vật lý.

## 3. Nguy cơ theo tình huống

Giữ nguyên checkpoint và nhãn lịch sử của dataset. Runtime hiện đại dùng chính
sách riêng: đồ chơi/trái cây/thìa gần miệng đủ bằng chứng yêu cầu chú ý ngay sau
cửa sổ xác nhận, không chờ 3 giây vì nhãn `safe`. DANGER hiện chỉ áp dụng nhóm
vật sắc/nhọn đã liệt kê, đủ xác suất, mask, quan sát và thời gian. Ổ điện/máy sưởi
cần ngữ cảnh khác; không tự gán nguy cơ nghẹn từ tên vật hay kích thước ảnh.
CSV legacy vẫn giữ luật lịch sử, được ghi rõ phiên bản chính sách riêng.

Kiểm chứng: nhóm safe, ngoài phạm vi, unknown, mask yếu, replay và hồi quy cảnh báo.
Chính sách là định nghĩa thử nghiệm, chưa có xác nhận y khoa hoặc đo hiệu quả thật.

## 4. Metric thống nhất và chất lượng ảnh

Report, grid và tradeoff mặc định dùng chung `explicit-exposure-v2`, duration khai
báo đầy đủ và chỉ các đoạn `non_event` làm mẫu số FA/giờ. Alarm kéo dài qua đoạn
dương và âm vẫn bị tính cảnh báo nhầm. Grid/tradeoff chỉ chạy dev. Tinh chỉnh từ
CSV chỉ thay tham số cảnh báo; thay detector/crop cần thu phiên mới.

Lệnh cũ dùng `--legacy-metrics` để đọc báo cáo lịch sử với cảnh báo không so sánh
trực tiếp. Không dùng đường legacy để chọn tham số cho báo cáo sản phẩm.
Thêm `image_observation_coverage` đo khả năng nhìn rõ ảnh theo gate hiện tại;
`observation_coverage` vẫn là độ phủ nghiêm ngặt bao gồm hiệu chuẩn/phạm vi model.
Chất lượng ảnh tốt không tự chứng nhận model hoặc tạo trạng thái xanh.

Kiểm chứng: metric qua cùng dữ liệu dev/report, unknown exposure, không tune test,
coverage không phụ thuộc cờ hiệu chuẩn, CLI report frozen-config và replay.

## 5. Định vị và gán vật

Khoảng gán tay–vật dùng tỷ lệ kích thước bàn tay, không dùng bán kính 80 pixel cho
mọi độ phân giải. Vật có nhiều chủ thể ứng viên chỉ hạ chất lượng những người đó.
Tách module `vision.objects` chứa gán vật và bộ bằng chứng mask theo từng người.
Mask kiểm tra phần nằm trong box, thành phần liên thông trội và ổn định hình dạng/
diện tích qua thời gian. Giai đoạn chờ xác nhận vẫn có CAUTION; không tích lũy
DANGER từ một mask mới hoặc đột biến. Các ngưỡng đều chưa hiệu chuẩn dev thực tế.

Kiểm chứng: cùng cảnh ở bốn tỷ lệ ảnh, người thứ ba độc lập, vật dài, mask ngoài
box/rời rạc/đột biến, mất mẫu và nhất quán replay. Không suy ra tiếp xúc vật lý từ 2D.

## 6. Ngân sách xử lý và đo đầy đủ

Giới hạn mặc định một lần YOLO crop phụ mỗi frame, luân phiên người để không bỏ
đói người cuối. Detector toàn khung vẫn chạy mỗi frame hợp lệ. Khi hết deadline
không khởi tạo/chạy thêm MediaPipe/classifier/ROI. Log số lần detector và thời gian
giải mã, báo mẫu hết hạn/vượt ngân sách, phân bố số người được theo dõi.

`profile_live_session.py` đo cả chuỗi camera tại máy bằng model thật, bao gồm cold
start, reconnect và thời gian không có frame; chỉ lưu CSV/JSON. Phải khai báo người
lớn đồng ý tham gia hoặc ma-nơ-canh. Không chạy camera tự động trong đợt sửa code.
Chưa có số đo 1/2/4 người hay recall thật; giới hạn ROI có thể ảnh hưởng recall và
cần phiên dev đối chứng trước khi chọn cấu hình triển khai.

Đo smoke cục bộ: CPU Torch 2.6.0+cpu, 2 threads, ảnh hình học 640×480 không có
người; 8 mẫu sau 2 warmup. Median/p90 (ms): YOLO 199,93/209,63; classifier
64,95/76,78; MediaPipe riêng 37,00/41,30; pipeline cảnh không người 193,63/219,74.
Pipeline này không chạy workload mốc của nhiều người, không phải FPS camera hay
độ trễ cảnh báo. Artifact số liệu tại `outputs/review_runtime_smoke_20261018.json`
(không commit output). Không cộng các percentile riêng thành latency toàn hệ thống.
