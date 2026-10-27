# ChildSafe Android — trạng thái phát triển

Nhánh: `codex/android-companion`. Bản đang phát triển, chưa sẵn sàng dùng để
giám sát trẻ. Không có kết quả kiểm chứng trên điện thoại hoặc ca nghẹn.

## Kiến trúc đã có

- `android/core`: mã hóa AES-256-GCM tương thích Python; trạng thái monitor tách
  nguy cơ, chất lượng quan sát và kết nối. Trạng thái tự hết hạn cả khi một yêu
  cầu mạng còn chờ; tắt tiếng/cooldown không đổi mức nguy cơ.
- `android/app`: giao diện Kotlin native theo bố cục đã duyệt, tạo phòng theo dõi,
  QR ghép đôi một lần trong hai phút, quét QR cục bộ, lưu thông tin ghép đôi bằng
  Android Keystore, dịch vụ theo dõi và notification channel có âm báo.
- `src/childsafe/companion`: relay FastAPI/SQLite lưu sự kiện mã hóa có giới hạn;
  quyền monitor và camera riêng; thu hồi camera; phiên mới khi kết nối lại;
  chặn số thứ tự lặp và tin hết hạn. Relay không nhận khóa giải mã hay video.
- `EventPublisher`: giao diện gửi sự kiện từ runtime Python, bốn cảnh báo chờ và
  một trạng thái mới nhất. Chưa nối vào vòng lặp webcam trong bản này.
- Công cụ xuất classifier/YOLO11n-seg ONNX, đóng gói MediaPipe và manifest SHA-256.
  Lớp `other` vẫn là thông tin chưa xác định trong runtime dù checkpoint cũ ghi
  binary_map là `safe`.

Hai thiết bị đều phải truy cập được relay. LAN có thể đặt relay trên máy tính
cùng mạng. Internet cần máy chủ HTTPS có thể truy cập từ ngoài, cùng cấu hình
thông báo đẩy cho Android; hiện chưa triển khai relay Internet hay FCM.
Không mở cổng Internet hoặc thay đổi firewall tự động.

## Phần chưa hoàn tất

1. Nối camera Android với ONNX/MediaPipe và gate cảnh báo; hiện ứng dụng nói rõ
   chưa bật nhận diện sau khi ghép đôi camera. Chưa được xem đây là camera hoạt động.
2. Đối chiếu tiền xử lý ảnh, mask, hướng mặt và diễn biến theo thời gian giữa
   Android với Python; chưa có kết quả tương đương toàn bộ pipeline.
3. Nối webcam Python vào publisher; kiểm tra hai thiết bị với TLS thực.
4. Thông báo Internet khi app bị hệ thống dừng/Doze: chưa có FCM. Foreground
   service đang dùng để thử truyền tin, chưa bảo đảm nhận khi khóa máy lâu.
5. Thử cài APK và đo hiệu năng trên điện thoại Android, mất mạng/khóa màn hình,
   quyền thông báo bị từ chối, đổi mạng, camera bị ứng dụng khác chiếm và nóng máy.
6. Cần thêm quản lý thiết bị/thu hồi quyền trong giao diện. API thu hồi đã có.
   Xóa cấu hình cục bộ hiện không tự thu hồi camera trên relay.

Relay hiện dành cho thử nghiệm có kiểm soát. Giới hạn tốc độ và số phòng giúp
chặn lạm dụng đơn giản, chưa thay thế đăng ký/xác thực người dùng và quản trị
dịch vụ công khai. Phòng không tự hết hạn; chủ phòng có API xóa phòng và dữ liệu.

## Xác minh đã thực hiện

- Python: 29 kiểm thử giao thức/client/relay qua. Bao gồm mã ghép đôi dùng một lần,
  đồng thời ghép đôi, phân quyền, chống lặp, thời gian hết hạn, reset phiên, mất
  kết nối, giới hạn lịch sử, mất đoạn lịch sử, hàng đợi và dữ liệu không chứa ảnh.
- Kotlin JVM: kiểm thử core biên dịch/chạy được với JDK 17, Gradle 8.11.1. Có mẫu
  AES-GCM công khai dùng chung với Python; đây không phải khóa ứng dụng.
- Lần kiểm tra mới nhất: 20 kiểm thử Kotlin core qua, bao gồm gate camera độc lập.
  Bộ Python đầy đủ: 308 qua, 3 bỏ qua. Các số này không phải số cảnh thử camera.
- Classifier có 11 lớp tại `outputs/with_other/runs/efficientnet_b0_group_robust/best.pt`:
  xuất ONNX opset 17. Trên ba tensor từ ảnh ngẫu nhiên với ba tỷ lệ khung khác nhau,
  sai lệch logits tuyệt đối lớn nhất so với PyTorch khoảng `3.1e-6` trên CPU Windows.
  Phép thử này không kiểm tra bước biến đổi ảnh native Android hay độ chính xác.
- Detector YOLO11n-seg xuất được ONNX `(1,116,8400)` và `(1,32,160,160)`;
  chưa đối chiếu số học detector hoặc hậu xử lý mask Android.
- Chưa có số FPS, độ trễ cảnh báo hai thiết bị hoặc recall trên điện thoại.

## Dựng và kiểm thử

Tất cả tệp công cụ, trọng số, SDK, khóa, cấu hình camera và bản dựng để trong các
đường dẫn được bỏ qua bởi Git. Không gửi video/âm thanh lên relay.

```powershell
rtk proxy uv pip install --python .venv/Scripts/python.exe -r requirements-companion.txt
rtk proxy uv pip install --python .venv/Scripts/python.exe -r requirements-android-export.txt
rtk proxy .venv/Scripts/python.exe -m pytest tests/test_companion.py tests/test_companion_client.py -q
rtk proxy .venv/Scripts/python.exe scripts/android_toolchain.py :core:test -PcoreOnly=true
```

Trên máy hiện tại, người dùng đã đồng ý điều khoản Google để cài SDK vào
`outputs/android-tools/sdk`. Mỗi người dựng dự án trên máy khác phải tự đồng ý
điều khoản trước khi dùng `scripts/install_android_sdk.py`; không mặc định chấp
nhận thay người dùng mới.

```powershell
rtk proxy .venv/Scripts/python.exe scripts/export_android_models.py --checkpoint outputs/with_other/runs/efficientnet_b0_group_robust/best.pt --detector models/yolo11n-seg.pt --face models/face_landmarker.task --hand models/hand_landmarker.task
rtk proxy .venv/Scripts/python.exe scripts/package_android_models.py
rtk proxy .venv/Scripts/python.exe scripts/android_toolchain.py :app:assembleDebug
```

Android tối thiểu dự kiến: Android 8 (API 26). Chưa xác nhận tương thích thiết bị.

## Relay LAN và chứng chỉ

Chọn đúng IP LAN của máy chạy relay; IP trong ví dụ phải thay bằng IP thực.
Tạo chứng chỉ mới sẽ đổi fingerprint: cần cấu hình lại các thiết bị.

```powershell
rtk proxy .venv/Scripts/python.exe scripts/companion_certificate.py --host 192.168.1.10
rtk proxy .venv/Scripts/python.exe scripts/companion_relay.py --host 0.0.0.0 --cert outputs/companion/tls/cert.pem --key outputs/companion/tls/key.pem
```

Nhập `https://192.168.1.10:8443` và SHA-256 fingerprint được in trên máy relay vào
máy theo dõi. Chỉ lấy fingerprint trực tiếp từ máy của nhóm. Kiểm tra hostname
vẫn được giữ; không có tùy chọn bỏ qua TLS. Tệp khóa riêng ở máy relay.
Sau đó tạo QR trên máy theo dõi, dùng máy camera quét trong hai phút.

Chỉ thử bằng đạo cụ an toàn và người tham gia đồng ý; không diễn lại nghẹn hay
đưa đồ vật nguy hiểm vào miệng. Kết quả là cảnh báo hỗ trợ quan sát, không chẩn đoán.
