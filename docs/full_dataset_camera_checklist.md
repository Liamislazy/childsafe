# Demo model đầy đủ và checklist camera góc cao

> Các số và hành vi dưới đây là lịch sử kiểm tra 08/10. Bản 12/10 yêu cầu mask
> định vị đáng tin cậy cho DANGER; hand crop chỉ CAUTION, gate quality dùng pixel
> gốc. Xem [hướng dẫn hiện tại](product_runtime_guide.md); checkout hiện tại chưa
> có checkpoint full-data và launcher dùng robust + other khi chạy mặc định.

Checkpoint Kaggle đã được kiểm tra ngày 08/10/2026: EfficientNet-B0, 11 lớp, letterbox 224×224, temperature 1,7798. Train 16 epochs, checkpoint tốt nhất có `best_epoch=9` (lượt thứ 10). Test tổng hợp: accuracy 92,63%, unsafe recall 90,87%. Open Images test độc lập: accuracy 91,99%, unsafe recall 92,19%. Đây là kết quả phân loại ảnh; camera góc cao chưa được nghiệm thu.

Kiểm tra tích hợp local sau sửa ảnh bút: 81 tests passed, 1 skipped (test detector với ảnh được duyệt riêng chưa được bật), 3 cảnh báo dependency deprecation. Test mới sử dụng checkpoint thật qua API ảnh/frame, xác minh letterbox, temperature và unknown rejection; kiểm tra pipeline thật và CSV cùng các chuỗi giả lập danger/caution/mất quan sát/reset qua API web. SHA-256 checkpoint: `76bb3a5f0928704c984de5150ac8cbf40157c25608c6d5374ba09c7a8d36d439`.

Benchmark CPU trên ảnh hình học tổng hợp, 10 mẫu: classifier median khoảng 36 ms, pipeline không có người khoảng 88 ms. Không dùng các số này làm FPS camera nhiều người. Báo cáo local tại `outputs/full_dataset_results/runtime_benchmark.json`; chưa có event recall, FA/giờ hoặc latency camera thực tế.

## 1. Chuẩn bị và mở demo

Trong checkout hiện tại, ZIP đã được giải nén vào `outputs/full_dataset_results/`. Giữ model và kết quả ngoài Git. Trên máy/checkout khác, chạy từ thư mục dự án:

```powershell
Expand-Archive -LiteralPath 'C:\duong-dan\full_dataset_results.zip' -DestinationPath outputs\full_dataset_results
```

Checkpoint phải nằm ở `outputs/full_dataset_results/efficientnet_b0_robust_letterbox/best.pt`, không có thêm thư mục bọc ngoài. Nếu kết quả cũ đã tồn tại, dùng thư mục mới và truyền đường dẫn checkpoint rõ ràng. Không dùng `kaggle_full.yaml` cho runtime local: nó chứa đường dẫn dữ liệu trên Kaggle.

Môi trường Python 3.12 cần các dependency trong `requirements-runtime.txt` và `pip install -e .`. Trong checkout hiện tại, `.venv` đã được nối với dependency local đã có; không cần tải lại. Model MediaPipe local nằm tại `models/face_landmarker.task`, `models/hand_landmarker.task`; detector tại `models/yolo11n-seg.pt`.

```powershell
.venv\Scripts\python scripts\web_demo.py
```

Mở `http://127.0.0.1:8000`. Live webcam dùng checkpoint đầu tiên (`full dataset (Kaggle)` mặc định) qua đầy đủ pipeline người, tay, miệng, mask vật và cảnh báo riêng theo ID. (Tab Photo test đã được gỡ; demo chỉ còn camera trực tiếp.) `other`, vật không nhận dạng và confidence thấp phải trả `uncertain`, kể cả khi P(unsafe) thấp. Letterbox và temperature được lấy tự động từ checkpoint.

Có thể mở server nền trên Windows bằng `powershell -ExecutionPolicy Bypass -File scripts\start_web_demo.ps1`. Dừng server đã mở bằng script này với `powershell -ExecutionPolicy Bypass -File scripts\stop_web_demo.ps1`. Log khởi động nằm ở `outputs/web_demo/`. Chỉ mở một server trên mỗi port; đóng server cũ trước khi khởi động lại.

Bấm Start camera, kiểm tra khung người có ID, điểm tay/miệng và mask vật. Bảng trạng thái nêu rõ thiếu thành phần nào; xám “CHƯA ĐỦ QUAN SÁT” không có nghĩa an toàn. Bấm Thử âm báo để kiểm tra loa. Âm ngắn xuất hiện khi cần cảnh giác hoặc nguy hiểm; có thể bỏ chọn Âm báo. Download số liệu phiên trước Reset alerts hoặc trước Start camera phiên mới: dữ liệu chỉ giữ trong RAM, gồm features/timestamp/ID và không gồm ảnh. Đổi tên `camera_session.csv` theo mã phiên để dùng trong đánh giá bên dưới.

Đũa/bút chưa có detector riêng. Khi thấy rõ tay và miệng, tay gần miệng có thể tạo CAUTION. Mask vật được gán cho tay, confidence ≥0,5 nhưng dưới ngưỡng xác nhận 0,6, sát vùng miệng cũng tạo CAUTION; không cộng dwell hoặc xác nhận DANGER từ mask này. Nhãn detector có thể sai (bút trong ảnh kiểm tra bị gọi là toothbrush). Nếu không có mask và tay xa, đầu vật dài vẫn có thể bị bỏ sót: cần detector riêng với dữ liệu gán nhãn. Thử luồng nguy hiểm bằng đạo cụ mô phỏng lớp đã có, không dùng vật sắc thật. Nếu FPS thấp, không hạ ngưỡng liên tục để che vấn đề.

Sửa lỗi từ ảnh thực tế ngày 08/10: một người được nhận diện tay trên toàn khung thay vì crop sát thân; landmark nằm ngoài khung người không được gán cho người đó. ID/model được giữ tối đa 3 giây để tránh khởi tạo lại liên tục khi xử lý chậm, còn khoảng liên tục để tích lũy nguy hiểm vẫn 0,25 giây. ID đã hết hạn được loại khỏi bảng người. Kiểm tra lại ảnh được cung cấp tại 640×480: một người, một tay, CAUTION vì vùng vật nghi ngờ sát miệng, dwell=0; năm lần xử lý với cả khoảng cách frame 1,3 giây vẫn giữ ID. Đây là kiểm tra ảnh tĩnh, chưa xác nhận chất lượng webcam liên tục.

## 2. Chạy pipeline camera để thử cảnh báo

```powershell
.venv\Scripts\python scripts\run_webcam.py --camera 0 --log-features --condition elevated_dev_s01
```

Model Kaggle mới là mặc định; có thể đổi bằng `--ckpt`. Phím `q` thoát, Space tạm dừng hiển thị. Giữ `quality.calibrated=false` và `detector.complete_object_coverage=false` cho đến khi có kiểm chứng. Detector pretrained chưa bao phủ bút/đồng xu.

App xử lý local và không ghi video. `--log-features` ghi CSV chứa timestamp, ID tạm thời, tọa độ và trạng thái; event CSV ghi cảnh báo. Thử với người lớn đồng ý hoặc búp bê, đạo cụ tù/lớn, chỉ đưa gần vùng miệng, không cho vào miệng. Người điều khiển phải dàn cảnh trực tiếp; lượt kiểm tra tự động không thay thế phần này.

## 3. Checklist mỗi phiên

- [ ] Ghi ngày, người thực hiện, dev/test, vị trí/góc camera, khoảng cách, độ phân giải, ánh sáng, số người, đạo cụ và mã phiên.
- [ ] Ghi SHA-256 của checkpoint và giữ bản config dùng trong phiên.
- [ ] Kiểm tra tên model, crop đồ vật, trạng thái uncertain của vật lạ trước khi chạy cảnh.
- [ ] Kiểm tra từng người có ID riêng; người ra/vào hoặc giao nhau không dùng lại lịch sử cảnh báo sai.
- [ ] Đánh dấu thời điểm bắt đầu/kết thúc từng sự kiện theo cột `t` trong feature CSV, không dùng giờ đồng hồ khác.
- [ ] Ghi cả khoảng không có sự kiện, mất mặt/tay, che khuất và mất frame; không bỏ các đoạn khó khỏi mẫu số.
- [ ] Lưu kết quả vào bảng bên dưới; chưa chạy ghi “chưa thử”, không ghi “đạt”.

## 4. Ma trận thử ban đầu

Mỗi cảnh có sự kiện thử tối thiểu ba lần; đây là bước khảo sát dev, chưa đủ kết luận triển khai. Thêm khoảng nền không có sự kiện đủ dài để đo báo nhầm; luôn báo tổng thời gian thực tế.

| Mã | Cảnh/điều kiện | Kiểm tra chính | Trạng thái |
|---|---|---|---|
| D01 | Camera ngang, một người, sáng, 720p | Mốc so sánh; vật gần/xa miệng | Chưa thử |
| D02 | Góc cao, một người, mặt thẳng | Vùng miệng và mask vật đúng vị trí | Chưa thử |
| D03 | Góc cao, mặt nghiêng/quay lưng | Mất quan sát phải thể hiện giới hạn | Chưa thử |
| D04 | Góc cao, ánh sáng yếu/nhòe | Không suy ra an toàn từ ảnh kém | Chưa thử |
| D05 | Góc cao, 480p/720p/1080p | Vật nhỏ, đầu vật dài không bị mất crop | Chưa thử |
| D06 | Hai/bốn người, giao nhau, đổi vị trí | Đúng người, không chuyển lịch sử | Chưa thử |
| D07 | Vật giữa hai người, nhiều vật một người | Liên kết mơ hồ không tạo danger giả | Chưa thử |
| D08 | Tay che miệng ≤0,8s rồi che lâu hơn | Không cộng dwell từ tọa độ giữ lại | Chưa thử |
| D09 | Vật dài gần miệng nhưng tay xa | Khoảng cách dựa trên mask vật | Chưa thử |
| D10 | Bút/đồng xu mô phỏng, vật lạ, thiếu mask | Ghi nhận giới hạn detector/uncertain | Chưa thử |
| D11 | Gãi mũi, chống cằm, chạm mặt, ăn/uống mô phỏng | Đo cảnh báo nhầm | Chưa thử |
| D12 | Người rời khung, nguồn ngắt/kết nối lại | Reset lịch sử, không giữ màn hình xanh/cảnh báo cũ | Chưa thử |

Sau khảo sát, cố định bộ cảnh test với người, vật và phiên khác dev. Không thay ngưỡng sau khi xem test.

## 5. Ghi nhãn và đánh giá

Sao chép feature CSV sau mỗi phiên vào `outputs/features/<ma-phien>.csv`; giữ tên này trong `video` của hai bảng theo [schema annotations](../data/scenarios/README.md). Đây cũng là mã phiên khi không lưu video. Bảng annotations có thể thêm `person_id,session_id` từ features để chấm đúng người và nguồn reconnect. Dùng nhãn `unsafe_mouthing` cho sự kiện đạo cụ nguy hiểm tiến sát vùng miệng đã quy định; không thực hiện việc đưa vật vào miệng.

| Phiên | Dev/test | Sự kiện thật | Bỏ sót | Báo nhầm | Thời gian quan sát | Recall | FA/giờ | Latency median/p90 | FPS | Coverage | Lý do lỗi |
|---|---|---:|---:|---:|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | … | … | … | … | … |

```powershell
.venv\Scripts\python scripts\evaluate_alerts.py report --split dev
.venv\Scripts\python scripts\evaluate_alerts.py grid --split dev
.venv\Scripts\python scripts\evaluate_alerts.py report --split test --config outputs\tuned_config.yaml
```

Chỉ chạy khi đã có features, annotations và `videos.csv` chia dev/test theo người. Chọn mục tiêu recall/FA mỗi giờ trước khi tuning; `--min-recall` là mục tiêu dev, không phải chứng nhận an toàn. Grid điều chỉnh rule, không tự hiệu chuẩn các proxy quan sát. Báo cáo recall tổng cùng coverage, không chỉ recall trên phần dễ quan sát. Latency cần đối chiếu timestamp sự kiện thật; `inference_s` chỉ là thời gian xử lý.

Chọn dữ liệu bổ sung từ lỗi dev; không đưa test trở lại train. Nếu bút/đồng xu là yêu cầu, cần detector và nhãn riêng trước khi tuyên bố bao phủ chúng. Chi tiết runtime tại [elevated_camera.md](elevated_camera.md).
