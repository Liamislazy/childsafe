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

## 7. Thu dữ liệu đúng lỗi và đóng phiên có thể kiểm toán

Checklist tạo 20 điều kiện/split, kết quả trống; bao phủ góc cao, vật dài/che khuất,
đổi người, thao tác thường, ngoài phạm vi và stress 4 người. Không tải dataset mới.
`finalize_session.py` ghép ba export số từ trình duyệt với manifest đã có consent/
chia subject; kiểm tra epoch, provenance và thời gian trước khi ghi. Đóng băng
checksum feature/annotation, report phát hiện thay đổi nội dung sau đó.

Hướng dẫn `targeted_camera_trials.md` yêu cầu đạo cụ lớn/mềm, không đưa vào miệng,
ground truth do người quan sát độc lập. Không có dữ liệu camera mới trong đợt này.
Kiểm chứng: kế hoạch tái lập, finalize/report, phát hiện sửa dữ liệu, marker chưa
đóng, model khác và frame sau khi kết thúc. Toàn bộ ví dụ kiểm thử là số liệu giả.

## 8. Ranh giới xử lý theo người

Tách crop/chuyển tọa độ/loại mốc ngoài người vào `vision.people`, gán vật/mask vào
`vision.objects`, chính sách nguy cơ vào `alerts.hazards`, công bố phiên web vào
hàm riêng và báo cáo/tuning vào module chung. Bổ sung kiểu dữ liệu cho feature và
trạng thái người; CSV vẫn là record thuần, đọc được dữ liệu lịch sử.

MediaPipe pooled instance dùng IMAGE mode không mang tracker VIDEO nội bộ sang
ID khác. Bộ lọc/hysteresis của dự án vẫn nằm riêng theo người. IMAGE có thể chậm
hơn khi tracking VIDEO hoạt động tốt; cần đo lại trên camera thật. Đây là thay đổi
để tránh lịch sử model ẩn vượt qua việc reset lịch sử logic, không phải tăng FPS.

## Kiểm chứng cuối đợt

- Toàn bộ Python suite: **205 passed, 3 skipped**, 17 deprecation warnings, 24,66 s.
  Hai notebook Kaggle đã tạo lại từ source/config mới; không chạy train.
- Test tích hợp trọng số thật trên crop đồ vật đã duyệt: **1 passed** (chạy opt-in
  riêng). Không có người hoặc video trẻ trong fixture.
- Edge headless với canvas stream: bật/tắt, marker độc lập và CSV hoạt động; không
  lỗi JavaScript. Hồi quy phản hồi renew cũ bị giữ lại qua stop/start thành công;
  phiên mới không bị ghi đè/tắt. Script `tests/browser_session_races.cjs` có thể
  chạy với Playwright đã cài và server local; không gọi camera/micro thật.
- `uv pip check`: 79 package hiện có tương thích. Chưa thử cài môi trường sạch/GPU.
- Smoke IMAGE mode model thật: 8 mẫu sau 2 warmup, cùng ảnh hình học không người,
  CPU 2 threads. Median/p90 ms: YOLO 194,20/212,47; classifier 66,65/78,52;
  MediaPipe riêng 37,34/40,83; pipeline không người 188,39/211,22.
  Artifact `outputs/review_image_mode_smoke_20261018.json`, không commit output.

Những số đo smoke không phải camera FPS với người, recall, FA/giờ hoặc latency
sự kiện. Dữ liệu thực tế, hiệu chuẩn, khả năng tổng quát sang trẻ và nhận diện
nghẹn vẫn chưa được kiểm chứng. Không tải dataset lớn hoặc train lại model.
