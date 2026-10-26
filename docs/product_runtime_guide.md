# Runtime sản phẩm: chạy, thử và báo cáo

Cập nhật 18/10/2026. Phạm vi: vật nguy hiểm tiến sát vùng miệng trên ảnh, một
camera, các người có ID tạm thời. Không xác nhận tiếp xúc vật lý, nuốt hay nghẹn.
Chưa hiệu chuẩn camera thật; bút/đũa/đồng xu chưa có detector riêng. Không tự đổi
`calibrated`/`complete_object_coverage` để giao diện hiện xanh.

## Chạy trên máy hiện tại

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_web_demo.ps1
```

Mở `http://127.0.0.1:8000`, chọn **Bật camera**. Một tab sở hữu phiên camera;
tab khác không được trộn frame hoặc reset phiên đó. Camera/micro không tự bật
khi mở trang; đợt này không thu âm. Ảnh trong RAM, không ghi video.

Launcher chọn checkpoint local đầu tiên tồn tại. Mặc định là MobileNetV3-Large
(10 lớp ChildSUn, chia theo nhóm, không có lớp "other"); robust + other chỉ còn là
dự phòng khi thiếu checkpoint MobileNet. `/api/info` cho biết model
thực tế, hash config/classifier/detector và phạm vi kết quả. Model cần đặt local;
không tự tải khi thiếu. Kết quả NORMAL khi thiếu quan sát có màu xám, không phải safe.
Khi chưa có nguy cơ và ảnh không cho thấy rõ miệng (`image_observation=unobservable`),
thẻ người hiện thanh xám **KHÔNG KẾT LUẬN · camera không nhìn rõ** thay cho NORMAL.
Nhãn YOLO `pen` là lớp `toothbrush` của COCO được đổi tên (COCO không có lớp bút); đây
không phải detector bút và nĩa hay vật mảnh khác cũng có thể hiện nhãn này.
Dao hoặc kéo mà YOLO gắn với bàn tay một người (độ tin cậy ≥ 0,6; trong 0,6 s gần nhất có
ít nhất hai khung và ít nhất một nửa số khung thấy dao/kéo, kéo dài ≥ 0,2 s) được báo
**NGUY CƠ CAO** ở bất kỳ vị trí nào, không cần gần miệng (`rules.held_sharp_*`); phát hiện
yếu hơn chỉ báo cần chú ý. YOLO hay nhầm dao với kéo nên lý do ghi "knife or scissors".
Khi tay nắm cán dao, MediaPipe thường không thấy bàn tay (0/113 khung trong một lần thử
webcam); lúc đó dao/kéo nằm ít nhất một nửa trong khung của đúng một người không có tay
quan sát được cũng được tính là đang cầm. Vì vậy dao trên bàn nằm trong khung người, khi
không thấy tay, cũng có thể bị báo. Dao/kéo nằm ngoài khung người không kích hoạt quy tắc
này. Ngưỡng chưa hiệu chỉnh.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1
```

CLI trực tiếp, ghi số liệu khi yêu cầu:

```powershell
.venv\Scripts\python scripts/run_webcam.py --camera 0 --ckpt outputs/with_other/runs/efficientnet_b0_group_robust/best.pt --log-features --condition elevated_dev
```

URL camera vẫn qua `--source-env`, không ghi URL/mật khẩu vào command/Git/manifest.
Hàng đợi một slot ưu tiên ảnh mới, log số lần slot bị ghi đè. Giới hạn tuổi ảnh
kiểm tra trước inference và trước xuất bằng chứng; frame hết hạn không tăng dwell.
Nguy cơ cũ chỉ được giữ ngắn theo cấu hình, sau đó kết thúc do mất quan sát.

## Chuẩn bị một phiên dev/test

Chỉ thử với người lớn đồng ý hoặc ma-nơ-canh, đạo cụ tù/lớn và không đưa vật vào
miệng. Người lớn đồng ý tham gia không đồng nghĩa trường đã duyệt quy trình;
hỏi giáo viên về yêu cầu của trường trước khi thu dữ liệu đánh giá. Không dùng
video trẻ thật. Observer ghi nhãn riêng, không lấy mức cảnh báo làm ground truth.

```powershell
.venv\Scripts\python scripts/scenario_session.py --id dev01 --split dev --subject adult01 --condition high_720p_bright --ckpt outputs/with_other/runs/efficientnet_b0_group_robust/best.pt --consent-confirmed
```

Thư mục `outputs/scenarios/dev01` chứa manifest, config đóng băng và hai template
nhãn. Subject là mã ẩn danh ổn định; không ghi tên/số điện thoại. Tạo test bằng
người khác; công cụ từ chối cùng subject ở dev và test. `--kind doll` dành cho
ma-nơ-canh. Không xem ma-nơ-canh như bằng chứng chuyển giao sang trẻ.

Trong web, observer chọn tình huống rồi **Đánh dấu bắt đầu/kết thúc**. Các nút chỉ
ghi timestamp và nhãn, không thay đổi nguy cơ. Nhãn có ba giá trị:

- `unsafe_mouthing`: sự kiện đạo cụ tiến sát vùng miệng theo tiêu chí dàn cảnh
  đã thống nhất; không nhãn nghẹn hoặc xác nhận tiếp xúc thật.
- `non_event`: khoảng observer đã xác nhận không có sự kiện mục tiêu.
- `uncertain`: chưa rõ; không đưa vào thời gian âm tính.

Kết thúc nhãn trước khi tắt camera. Nhãn chưa kết thúc không được tự điền end
hoặc đưa vào đánh giá. Tải số liệu, nhãn và thông tin phiên **trước khi reset/bật
phiên mới**. Lưu số liệu thành `features.csv`, nhãn thành `annotations.csv` trong
thư mục phiên. Từ `session.json`, điền `start_s,end_s` vào `manifest.json`; giữ
`annotation_scope=camera` nếu không có mapping ID độc lập đáng tin cậy.

Một manifest chứa đúng một runtime epoch; reset/reconnect cần phiên riêng và
phân tách CSV theo session_id trước khi báo cáo. Với scope `person`, annotations
có thêm `person_id`; người không được phát hiện vẫn phải được ghi nhãn và vào
mẫu số recall. Không suy ra danh tính thật từ ID tracker; camera-level annotation
vẫn hữu ích khi mapping không xác minh được.

Timestamp nhãn là lúc server nhận thao tác observer; độ trễ thao tác con người
chưa đo. Frame browser dùng thời điểm snapshot, không phải camera sensor capture.
Độ trễ event báo cáo không được gọi là độ trễ lâm sàng/cảm biến chính xác.

## Hiệu chuẩn dev và đóng băng test

Observer ghi `observation_labels.csv` với `session_id,person_id,t,mouth_observable`
(0/1), độc lập với cờ của model. Có cả mẫu thấy rõ và không thấy rõ, timestamp
khớp feature. Chỉ dùng dev:

```powershell
.venv\Scripts\python scripts/calibrate_observation.py outputs/scenarios/dev01
```

Đầu ra `outputs/quality_proposal/proposed_config.json` là **đề xuất** min-face và
sharpness trên dev. Không hiệu chuẩn jitter/che khuất/tất cả góc; `calibrated=false`
được giữ. JSON được đọc trực tiếp để không đổi số khoa học qua YAML, hash config
được kiểm tra. Dùng cấu hình đề xuất cho các phiên dev mới và xem lỗi, sau đó
chốt ngưỡng trước khi thu test mới; không tune bằng kết quả test.

Mỗi phiên test phải tạo bằng đúng config/model chạy thực tế. CSV chứa hash runtime;
report từ chối hash khác manifest hoặc mã nguồn khác phiên đã thu. Giữ revision
mã nguồn của phiên để replay; không sửa config.json sau khi đã tạo phiên.
Model lỗi/thiếu hoặc metadata cũ thiếu provenance không được giả vờ là phiên đã
kiểm chứng; dùng evaluator legacy hiện có với giới hạn rõ.

## Báo cáo

```powershell
.venv\Scripts\python scripts/report_product_sessions.py outputs/scenarios --split dev --out outputs/product_dev.json
.venv\Scripts\python scripts/report_product_sessions.py outputs/scenarios --split test --out outputs/product_test.json
.venv\Scripts\python scripts/report_runtime_profile.py outputs/scenarios/dev01/features.csv
```

Báo cáo mới có recall tổng/quan sát được, CAUTION-or-DANGER riêng, FA trên thời
gian âm tính rõ nhãn, latency median/p90, coverage theo toàn thời gian khai báo,
baseline và ablation smoothing/temporal/object gate. Một alarm kéo dài vào đoạn
âm tính vẫn bị tính báo nhầm dù nó cũng trùng sự kiện dương tính. Unknown không
được tính âm tính. Khoảng tin cậy Wilson/Poisson được ghi cùng giả định; các lần
diễn lặp có thể không độc lập. Không dùng 0 báo nhầm trong đoạn ngắn để hứa không
báo nhầm. Test không có sự kiện/âm tính trả metric null thay vì số hoàn hảo.

Coverage `observable` bao gồm cả giới hạn model/calibration, nên có thể bằng 0
ở cấu hình chưa hiệu chuẩn. Không bỏ các đoạn đó khỏi recall tổng. Test legacy
thiếu quality trả coverage không xác định. Ablation ROI/detector cần các phiên mới
đối chứng tương ứng; replay chỉ thay rule, không tái tạo detector khác từ CSV.

Từ bản rà soát 18/10, báo thêm `image_observation_coverage`: tỷ lệ ảnh có miệng/tay
đủ rõ theo gate hiện tại, độc lập với cờ hiệu chuẩn và phạm vi lớp vật. Đây không
phải xác nhận mô hình đúng hoặc không có che khuất. `observation_coverage` vẫn giữ
định nghĩa nghiêm ngặt; cả hai dùng toàn thời gian phiên làm mẫu số.

Report, grid và tradeoff dùng chung phiên bản metric `explicit-exposure-v2`:

```powershell
.venv\Scripts\python scripts/evaluate_alerts.py grid --sessions outputs/scenarios --split dev --out outputs/grid_dev.json
.venv\Scripts\python scripts/evaluate_alerts.py tradeoff --sessions outputs/scenarios --split dev --out outputs/tradeoff_dev.json
.venv\Scripts\python scripts/evaluate_alerts.py report --sessions outputs/scenarios --split dev --config outputs/grid_dev_config.json --out outputs/tuned_dev.json
```

Chốt cấu hình đề xuất trước khi thu các phiên test mới. Report test dùng config
được đóng băng trong từng phiên; không dò nhiều ngưỡng trên test. Bộ đọc kiểm tra
subject dev/test, config, model và mã nguồn như lệnh report trước đây. Grid cần
có cả sự kiện dương và đoạn âm rõ nhãn, cùng cấu hình thu dữ liệu.
Đường CSV/báo cáo lịch sử của `evaluate_alerts.py` chỉ được bật bằng
`--legacy-metrics`; số từ đó có định nghĩa khác và không được ghép với báo cáo sản phẩm.

Timing gồm detector/landmarks/classifier/quality/other, peak memory toàn process,
processed FPS/gap và slot overwrites nếu CLI. Không đếm mỗi dòng người như một
frame. Memory không phải bộ nhớ riêng của một model. Warmup đánh giá hiệu năng
riêng, nhưng các sự kiện khởi động vẫn phải giữ trong recall toàn phiên.
Event CSV có session_id, lý do kết thúc và quality tại lúc bắt đầu sự kiện;
không dùng snapshot quality đó để suy ra coverage toàn sự kiện.

Đo toàn luồng tại máy trình diễn bằng người lớn đã đồng ý hoặc ma-nơ-canh:

```powershell
.venv\Scripts\python scripts/profile_live_session.py --source 0 --kind adult --consent-confirmed --condition elevated-720p-1person-bright --ckpt outputs/with_other/runs/efficientnet_b0_group_robust/best.pt --seconds 60 --width 1280 --height 720 --max-people 1 --cpu-threads 2 --out outputs/profile/elevated-720p-1person
```

Lặp lại cùng tình huống với 1/2/4 người và 480p/720p/1080p; ghi rõ camera có thực
sự trả độ phân giải yêu cầu hay không (kích thước ảnh gốc trong feature). Model,
thiết bị, cấu hình, timing từng bước và frame vượt ngân sách được lưu trong JSON.
Không lưu ảnh/video hoặc URL kết nối. Phiên đo hiệu năng này không có nhãn sự kiện;
không dùng nó để công bố recall/FA. Khi không có frame, thời gian thực vẫn được tính.
Luồng URL dùng `--source-env TEN_BIEN_MOI_TRUONG`; không dán URL có thông tin kết
nối vào lệnh, config, condition hoặc Git.

Mặc định chỉ một crop YOLO phụ/frame, luân phiên các người cần kiểm tra; toàn khung
vẫn chạy mỗi frame hợp lệ. Có thể so sánh `detector.roi_budget_per_frame=1` với `4`
trên các phiên dev đối chứng. Giới hạn công việc không chứng minh recall được giữ
nguyên. Giữ `stale_frame_s`/`max_gap_s` cho đến khi có đo dev, không tăng chỉ để làm
cho DANGER xuất hiện trên máy chậm.

## Nghiệm thu thực tế còn lại

Chạy ma trận trong `product_runtime_plan_to_2026_10_24.md`: camera ngang/cao,
mặt thẳng/nghiêng/quay lưng, 480p/720p/1080p, sáng/tối, 1/2 người, giao nhau,
che khuất, vật dài và ngắt nguồn. Bốn người là stress test. Đề xuất 60 sự kiện
test và hai giờ nền; ghi số thực tế nếu thiếu, không coi đây là chứng nhận an toàn.

Chưa có cảnh thử hợp lệ thì chưa có recall/FA/latency camera thực tế hoặc hiệu
quả ở trẻ. Bộ test tổng hợp, ảnh vật và luồng canvas chỉ kiểm chứng phần mềm.

Luồng crop theo người nằm trong `vision.people`; IMAGE mode của MediaPipe không
giữ tracker nội bộ giữa các người khi tái sử dụng model. Bộ lọc thời gian của dự án
vẫn riêng theo ID. Cần đo lại chi phí trên camera thật, đặc biệt bốn người. Quy trình
checklist và ghép export nằm trong `targeted_camera_trials.md`.
