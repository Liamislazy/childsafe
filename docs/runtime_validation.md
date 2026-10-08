# Kết quả triển khai runtime ChildSafe

Triển khai từ 07 đến 08/10/2026 (Asia/Saigon). Nhánh `codex/elevated-camera-implementation`, bắt đầu từ `6ddb27d` của worktree detached HEAD. Nhánh `codex/elevated-camera-runtime` đã tồn tại nên không ghi đè.

## Phạm vi đã triển khai

- Tách chất lượng quan sát khỏi nguy cơ; không xanh khi thiếu bằng chứng/model hoặc chưa hiệu chuẩn.
- Giữ nguy cơ ngắn hạn tối đa 0,8s; không tích lũy dwell từ landmark giữ lại; kết thúc sự kiện có lý do.
- YOLO11n-seg local, tối đa bốn người, crop MediaPipe riêng, quy đổi tọa độ gốc, lịch sử từng ID, loại lịch sử khi liên kết mơ hồ.
- Khoảng cách mask vật–vùng miệng; kiểm tra hướng mặt/kích thước/jitter/blur/exposure; box/tay gần đầu không xác nhận danger.
- Nguồn webcam/URL với slot một frame, loại frame trễ, timestamp nội bộ, reset khi reconnect, temporal theo giây và cooldown chỉ chặn notification.
- Schema v2, replay cùng bộ xử lý, coverage theo thời gian, recall tổng giữ sự kiện mất quan sát/người không phát hiện; legacy coverage không xác định.
- Khôi phục code/tài liệu Kaggle/supplement đã có trong checkout gốc; checkpoint preprocessing và unknown-class rejection.

## Commit

| Commit | Nội dung | Kiểm thử tại thời điểm commit |
|---|---|---|
| `41b8950` | fix: reject uncertain classifications and preserve brief occlusion evidence | 13 alert/regression tests |
| `dced7b6` | feat: separate observation quality from risk state | 16 targeted tests |
| `3e8946c` | feat: track people and associate hands and objects per person | 21 targeted tests |
| `a5b5174` | feat: evaluate visible object proximity for elevated camera views | 26 targeted tests |
| `0eff343` | fix: handle stale frames and make temporal alerts frame-rate independent | 30 targeted tests |
| `1475c2c` | feat: add reviewed supplemental data and Kaggle training workflow | 35 scoped tests, notebook sync |
| commit cuối của nhánh | test: add elevated-view scenarios and observation coverage reports | suite đầy đủ + integration model thật; xem kết quả bên dưới |

Workflow/preprocessing kế thừa được tìm thấy ở checkout gốc sau khi các commit runtime đã tạo, nên thứ tự thực tế khác vị trí thứ hai trong kế hoạch. Không sửa checkout gốc, không rewrite lịch sử nhánh của người dùng. Các commit có phạm vi độc lập và ghi kiểm thử/giới hạn trong message.

## Kiểm thử

Kết quả suite cuối: **61 passed, 2 warnings**, 15,18 giây, bao gồm integration model thật (không skip). `git diff --check` qua. Hai warnings là deprecation của protobuf trên Python 3.12.

Suite cuối gồm kiểm thử gốc, hồi quy category/binary confidence và unknown, mất landmark ngắn/dài, chưa thấy miệng/quay lưng, thiếu mask, vật dài–tay xa, crossing/ID/crop, model lỗi, cooldown, event end reasons, một-slot/reconnect, CSV roundtrip, live/replay theo người/phiên, coverage và sự kiện người không được phát hiện. Timing danger ở 5/10/15/30 FPS chênh lệch không quá một khoảng lấy mẫu tương ứng; đây là dữ liệu tổng hợp với smoothing tắt để tách riêng temporal.

Integration opt-in `tests/test_real_models.py` đã chạy với YOLO11n-seg, EfficientNet-B0 checkpoint và hai bundle MediaPipe thật trên crop dao đã được reviewed trong bộ bổ sung sẵn có. Xác nhận forward chạy, mask đúng kích thước ảnh, probabilities hợp lệ, không có face trên ảnh chỉ vật, pipeline không xanh và replay khớp. Không dùng ảnh/video trẻ em, không ghi ảnh hoặc video mới.

Lỗi notebook drift sau thay đổi config đã được phát hiện và sửa bằng tái tạo notebook. Các deprecation warnings của protobuf/MediaPipe trên Python 3.12 được giữ; chưa hỗ trợ Python 3.14.

## Benchmark đo thật

Đầu vào chính: crop vật đã reviewed `01b9021aa05e6e46.jpg`, 445×330, SHA-256 pixel RGB `a57e943824dd7a838d9269aaab597fc496db8fc2004aa1c861af94e1412d2f0f`. Không có người, nên pipeline chỉ chạy detector rồi trả unobservable; số pipeline FPS không đại diện một/bốn người. Warmup 2 vòng, đo 10 vòng, Windows 11 / CPU, PyTorch báo 8 threads trong phép đo. Thông tin model/hash trong `model_provenance.json`; phiên bản và raw timings ở `outputs/benchmark_object.json` (không commit artifact).

| Phần | Median (ms) | p90 (ms) | Throughput (FPS) |
|---|---:|---:|---:|
| YOLO11n-seg | 117,05 | 296,64 | 6,05 |
| EfficientNet-B0 | 68,43 | 177,42 | 11,67 |
| MediaPipe face + hand trên ảnh không người | 30,48 | 33,22 | 32,96 |
| Pipeline cảnh không người | 138,58 | 461,67 | 4,73 |

Ở các lần smoke trước, detector khoảng 12 FPS trên ảnh tổng hợp và 11,82 FPS trên cùng crop vật. Phép đo cuối biến động đáng kể; chưa xác định nguyên nhân, số mẫu nhỏ. Không dùng các số này để tuyên bố tốc độ webcam, độ trễ đầu cuối hoặc tốc độ bốn người. p90 pipeline vượt 0,25s là cảnh báo về khả năng duy trì temporal continuity trên CPU.

**Phát hiện thực tế:** YOLO gán nhãn crop dao thành `scissors` (confidence 0,871), mask 330×445 với 29.287 pixel. Classifier chọn `knife`, nhưng obj_conf=0,411 và P(unsafe calibrated)=0,559; logic từ chối thành uncertain. Mask hiện diện không chứng minh nhãn đúng hoặc mô hình đã phù hợp miền triển khai.

Không có annotation sự kiện webcam để đo recall thực tế, false alarms/giờ, latency sự kiện median/p90 hoặc coverage thực tế. Những số này chưa được đo, không thay bằng số synthetic. Report tooling đã có để đo khi thu được tập dev/test hợp lệ.

## Chưa nghiệm thu

Toàn bộ ma trận webcam ngang/góc cao × hướng mặt × 480p/720p/1080p × sáng/tối × một/hai/bốn người, che khuất, vật dài và nguồn URL chưa được chạy trên cảnh thử an toàn đã xác nhận. Chưa hiệu chuẩn các ngưỡng head pose/blur/exposure/size/jitter; vẫn `calibrated=false`. Chưa kiểm thử GPU. Chưa có detector tùy chỉnh pen/coin; pretrained không định vị các vật này. Chưa thực thi notebook Kaggle, chưa train mới, chưa tải thêm dataset. Phiên bản core đã kiểm thử cục bộ; chưa xác minh clean install toàn bộ dependency từ đầu.

Không push, không tạo PR; không commit dữ liệu ảnh/video, trọng số, camera URL hoặc credential. Không thể tuyên bố hoàn tất nghiệm thu thực tế hoặc sẵn sàng triển khai. Hướng dẫn thử và ma trận còn lại nằm trong `elevated_camera.md`.
