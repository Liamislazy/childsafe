# Kế hoạch hoàn thiện runtime cảnh báo nguy cơ đến 24/10/2026

Lập ngày 12/10/2026, từ `97022bc` trên `codex/choking-validation`; nền runtime
theo commit đồng nghiệp `88f40ac` và tối ưu `6cafaeb`. Đây là kế hoạch, chưa triển
khai các thay đổi bên dưới. Không push/PR, tải dataset lớn hoặc train mới trong
phần bắt buộc. Nhánh nhận diện nghẹn vẫn là mục tiêu riêng, chưa được kiểm chứng.

## 1. Kết quả sản phẩm cần có

Một camera webcam/URL, nhiều người, cảnh báo vật thuộc phạm vi đã kiểm chứng tiến
sát vùng miệng trên ảnh. Kết quả riêng theo người, tách nguy cơ khỏi chất lượng
quan sát. Không xác nhận nuốt/tiếp xúc vật lý/nghẹn, không chuyển 2D thành centimet.

Không hứa camera nhận diện được khi người quay lưng hoặc che hoàn toàn miệng:
đúng hành vi trong cảnh đó là thể hiện mất quan sát và vòng đời nguy cơ rõ ràng.
Giữ đầy đủ các sự kiện khó trong recall tổng. Dùng người lớn đồng ý và ma-nơ-canh,
đạo cụ tù/lớn, chỉ đưa gần vùng miệng, không đưa vật vào miệng. Không ghi video
mặc định, không dùng video trẻ thật, không gửi camera lên cloud.

## 2. Những gì đã có và khoảng trống đã kiểm tra

Đã có: tracker tới bốn người, crop và quy đổi tọa độ, mask proximity, trạng thái
quan sát/nguy cơ, giữ bằng chứng ngắn hạn, event end reasons, temporal theo giây,
webcam/URL một-slot/reconnect, replay và metrics. Không viết lại các phần này.

Khoảng trống cần sửa hoặc đo:

- `ElevatedQuality.measure` đang nhân kích thước mặt/miệng/tay với hệ số phóng to
  trước kiểm tra pixel. Crop giúp tìm mốc, nhưng phóng to không bổ sung chi tiết
  ảnh gốc. Ngưỡng quan sát phải dùng pixel gốc; ghi riêng scale cho chẩn đoán.
- Blur/exposure hiện đo toàn khung, có thể bỏ qua khuôn mặt tối/nhòe trước nền
  sáng/rõ. Cần đo ROI từng người/vùng mặt và không để lỗi một người ảnh hưởng người khác.
- Camera web gửi JPEG, runtime dùng thời điểm xử lý; không có timestamp/sequence
  capture đủ để kiểm tra ảnh lặp hoặc cũ. Slot một-frame của CLI không chứng minh
  đường browser có kiểm soát tuổi ảnh đầy đủ.
- Chưa có calibration dev góc cao hoặc test sự kiện thực tế. `calibrated=false`
  và `complete_object_coverage=false` phải giữ cho tới khi có chứng cứ đúng phạm vi.
- Detector pretrained chưa có lớp bút/đồng xu. Classifier có thể đặt tên vật nhưng
  không thay thế vị trí/mask đáng tin cậy. Detector gán bút thành toothbrush không
  chứng minh đã có detector bút. Bounding box chỉ là bằng chứng nghi ngờ.
- Checkpoint full-data được launcher ưu tiên nhưng thiếu trong checkout hiện tại;
  runtime dùng robust + other. Phải hiển thị model thật đang chạy và ghi hash.

## 3. Thứ tự triển khai và commit

### 12–14/10: độ tin cậy của quan sát

Commit: `fix: gate observation on original pixels and local image quality`.

- Giữ crop/zoom cho inference nếu hữu ích; kiểm tra face/hand/mouth extent trên
  tọa độ ảnh gốc. Đo local blur/exposure trên ROI gốc; không coi upsampling là
  phục hồi độ phân giải. Ghi original extents, crop bounds, scale và quality reasons.
- Thêm trường CSV có version/tương thích legacy để replay có cùng dữ liệu quality.
- Test cùng ảnh/mốc ở scale 1/2/3: kết quả gate kích thước không được tốt lên chỉ
  do phóng to. Test nền sáng–mặt tối, nền rõ–mặt nhòe, crop ngoài biên, nhiều người.
- Không tự đổi ngưỡng sau sửa code; chọn/hiệu chuẩn ngưỡng bằng dev ở bước sau.

### 14–16/10: độ mới của frame và phiên camera web

Commit: `fix: reject duplicate and expired browser camera frames`.

- Gửi session token, sequence và thời gian capture/browser metadata; dùng
  requestVideoFrameCallback khi hỗ trợ để phân biệt frame thật với lần vẽ lặp.
- Định nghĩa clock rõ ràng: browser monotonic không trừ trực tiếp server
  perf_counter. Muốn đo capture-to-server phải handshake offset/uncertainty;
  nếu chưa xác minh chỉ báo internal age/round trip, không gọi end-to-end latency.
- Chốt giới hạn tuổi/gap từ cấu hình; kiểm tra lại trước inference và trước dùng
  kết quả. Ảnh trễ/lặp/sai phiên không thêm dwell, không xác nhận nguy cơ mới.
- Stop/restart/reset loại callback/response cũ. Reconnect đổi phiên, không dùng
  lịch sử trước ngắt. Các client chia sẻ một singleton cần session ownership
  rõ ràng hoặc chỉ cho một phiên camera hoạt động; không trộn hai người dùng.
- Tests: ảnh đến sai thứ tự, timeout rồi response muộn, đổi phiên, browser pause,
  nhiều request, reset trong khi inference. Giữ không tích lũy queue và UI rõ lỗi.

### 16–18/10: hiệu năng trên tải người thật và hành vi vật chưa hỗ trợ

Commit: `perf: schedule live inference with measured freshness budgets`.

- Benchmark cùng model/config trên CPU hiện có; thử GPU riêng chỉ nếu môi trường
  phù hợp. Đo 1/2/4 người, không dùng benchmark ảnh vật để suy ra tốc độ nhiều người.
- Ghi thời gian detector/landmarks/classifier, processed frame gap, frame bị bỏ,
  median/p90 và peak memory. Warmup tách riêng; không chỉ báo FPS trung bình.
- Dựa vào bottleneck mới chọn giảm ROI pass, classifier interval hoặc input size;
  luôn so sánh recall/FA với cấu hình gốc trên dev. Không hạ ngưỡng để che FPS chậm.
- Nếu processed gap vượt max_gap_s, thể hiện thiếu liên tục; không nội suy nguy cơ
  qua khoảng trống. Detector còn cần mask mới đúng; cache phải có hạn sử dụng/ID.
- Bút/đồng xu chưa được định vị: hiển thị chưa hỗ trợ/không rõ, không xanh từ nhãn
  `other`/`safe`. Giữ mask thiếu/mơ hồ ở mức nghi ngờ. Kiểm tra vật dài–tay còn xa.

### 18–20/10: quy trình thử và hiệu chuẩn dev

Commit: `feat: add reproducible adult scenario sessions and dev calibration`.

- Chuẩn bị manifest phiên: người, dev/test, camera/góc, resolution, lighting, props,
  số người, config/model hashes. Tách người và phiên dev/test trước khi thử.
- Có đồng ý của người lớn; hỏi trường về quy trình thử trước khi thu recording.
  Mặc định log số/event và ghi nhãn trực tiếp cùng timeline, không lưu video.
  Ground truth do quan sát viên ghi độc lập; không lấy near/level của model làm nhãn.
- Định nghĩa sự kiện mục tiêu là đạo cụ tiến sát vùng miệng theo tiêu chí đã chốt,
  không dùng nhãn nghẹn. Ghi cả đối tượng không được phát hiện, mất quan sát và
  sự kiện bắt đầu trong lúc quay lưng. Không dùng tracker ID như danh tính thật;
  ghi mapping khi có thể, cảnh không map được vẫn phải vào recall ở mức camera.
- Khảo sát dev theo ma trận dưới; phân tích lỗi rồi tune rule/quality trên dev.
  Đóng băng config/hash trước test. Chỉ xác nhận calibration cho phạm vi đã đo;
  không bật một cờ toàn cục như thể mọi góc/mọi lớp đã được nghiệm thu.

### 20–22/10: test độc lập và báo cáo

Commit: `test: report condition-level event recall and observation coverage`.

- Mở rộng công cụ hiện có để nhóm theo condition và scope. Báo recall tổng, recall
  khi quan sát được, FA/giờ, số giờ nền, latency median/p90, FPS, frame gaps, coverage.
- Báo số sự kiện/mẫu số và khoảng tin cậy phù hợp; không gọi 0 alarm trong phiên
  ngắn là không có báo nhầm. Tính FA theo khoảng âm tính đã ghi nhãn, không bỏ toàn
  bộ đoạn khó hoặc coi đoạn chưa rõ nhãn là âm tính.
- Cùng diễn biến synthetic/replay tại 5/10/15/30 FPS, lệch chuyển trạng thái trong
  một khoảng lấy mẫu; thêm dropout thật, log roundtrip và reconnect. Phân biệt
  thử temporal tổng hợp với đánh giá model trên cảnh thật.
- So baseline distance-only và ablation ROI/cache/temporal trên cùng test. Không
  sửa ngưỡng từ lỗi test; thay đổi sau đó cần test mới độc lập.

### 23–24/10: đóng gói demo và tài liệu

Commit: `docs: publish measured runtime limits and science-fair demo guide`.

- UI tiếng Việt: nguy cơ + chất lượng quan sát riêng, rõ người nào bị cảnh báo,
  model/lớp đang hỗ trợ và nguyên nhân mất quan sát. Model thiếu/lỗi hiện hạn chế.
- Chọn cấu hình demo từ kết quả đã đo. Không hiện “safe” cho toàn cảnh chỉ vì
  lớp vật thuộc nhóm safe của dataset; diễn đạt giới hạn phạm vi quan sát.
- Hướng dẫn một lệnh chạy local, kiểm tra nguồn/hash/model, export report, reset
  và dừng; kiểm tra cài môi trường sạch. Không commit media, weights hoặc URL/mật khẩu.
- Kịch bản trình bày có cả cảnh phát hiện thành công và cảnh quay lưng/che khuất
  để chứng minh hệ thống thể hiện thiếu thông tin đúng. Điều kiện chưa thử ghi rõ.

## 4. Ma trận thử tối thiểu đề xuất

Không chạy full factorial mọi tổ hợp; chọn cảnh cơ sở rồi thay từng điều kiện,
thêm một số tổ hợp khó. Đề xuất 5 người lớn (3 dev, 2 test), ma-nơ-canh bổ sung;
số này là kế hoạch khả thi thời gian, không chứng minh tổng quát sang trẻ.

| Nhóm | Điều kiện |
|---|---|
| Góc camera | Ngang, cao; ghi góc/vị trí thực tế |
| Hướng mặt | Thẳng, nghiêng, quay lưng |
| Hình ảnh | 480p/720p/1080p; đủ sáng/thiếu sáng; nhòe chuyển động |
| Người | 1/2; 4 người là stress test, chưa mặc định phạm vi demo |
| Tương tác | Vật tới gần/rời miệng, vật dài–tay xa, đổi vật, vật lạ |
| Che khuất | Ngắn dưới 0,8 s, dài, tay che mặt, vật giữa hai người |
| Âm tính | Gãi mũi/cổ, chống cằm, chỉnh áo, nói chuyện, cầm vật xa mặt |
| Nguồn | Camera ngắt, reconnect, frame cũ/lặp, reset và model lỗi |

Mục tiêu thu ban đầu: ít nhất 60 sự kiện có nhãn test phân bố theo condition và
ít nhất 2 giờ nền âm tính test, báo số lượng thực tế nếu thiếu. Không xem đó là
đủ chứng nhận an toàn. Cần lưu riêng cảnh unsupported thay vì loại khỏi báo cáo.

## 5. Điều kiện nghiệm thu

- Tests liên quan qua từng commit; cuối đợt suite đầy đủ, kiểm tra model thật và
  launcher bằng local models. Giữ kết quả cũ làm baseline, không sửa test để che lỗi.
- Không nguy cơ mới từ tọa độ/frame/cache hết hạn; không chuyển lịch sử giữa người.
- Phóng to không nâng gate pixel; thiếu mask không thành xác nhận contact; unknown
  không thành safe. Cooldown chỉ giảm notification, không hạ risk đang tồn tại.
- Live và replay cùng quyết định từ cùng timeline/features; báo cả dữ liệu legacy
  thiếu quality. Chứng minh browser timestamp handling bằng các test clock/phiên.
- Có metrics cảnh thật theo condition, hashes/config đóng băng và miền vận hành
  rõ. Chọn mục tiêu recall/FA dev trước tuning; chưa đủ dữ liệu thì không nhận đạt.
- Không có đo thật thì chỉ nghiệm thu phần mềm, ghi chưa kiểm chứng hiệu quả camera.

## 6. Detector bút/đồng xu và âm thanh

Sau khi runtime/test nền ổn, có thể mở công việc detector riêng. Bộ Open Images
đã chuẩn bị là bounding box, không đủ train mask segmentation. Muốn DANGER dựa
trên vật dài gần miệng cần mask đã gán/kiểm tra hoặc phương pháp định vị đầu vật
được đánh giá riêng. Train detector box chỉ bổ sung CAUTION và nhãn định vị; không
âm thầm dùng box như mask thật. Chưa tải dataset lớn, chưa hứa hoàn tất trước 24/10.

Âm thanh chưa vào nhánh cảnh báo vật gần miệng đợt này. Timeline AV và evaluator
nghẹn giữ riêng; chưa có model/nhãn phù hợp nên không thêm “choking detected”.
Điều này không đổi mục tiêu nghẹn dài hạn; nó tách điều kiện nghiệm thu hai nhánh.
