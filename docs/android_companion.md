# ChildSafe Android — bản thử hai thiết bị

Nhánh `codex/android-companion`. Đã dựng được APK, nhưng **chưa xác nhận chạy và
cảnh báo đúng trên điện thoại thật**. Thiết bị dự kiến: Galaxy A03 làm camera,
POCO X7 Pro nhận cảnh báo. Chưa thử trên trẻ hay ca nghẹn; chưa sẵn sàng triển khai.

## Đã triển khai

- Một APK, hai vai trò. CameraX giữ ảnh mới nhất; MediaPipe tìm mặt/tay trên crop
  người và đổi về ảnh gốc; ONNX chạy YOLO11n-seg và classifier 11 lớp trên thiết bị.
  Kiểm tra checksum trước khi nạp; NNAPI có thể fallback CPU, giao diện nêu rõ.
- Tối đa bốn ID tạm thời, gate chuyển động/kích thước, bỏ lịch sử khi gán không
  chắc chắn. Không phải tái nhận dạng danh tính. Tay gần mặt đơn thuần không tạo
  cảnh báo. Hướng mặt, nhòe, sáng, kích thước gốc, mốc nhảy và tay phủ miệng làm
  giảm khả năng quan sát. Không xác nhận tiếp xúc, nuốt hay nghẹn.
- Đo mask với khoảng bất định do độ phân giải prototype; bbox chỉ tạo nghi ngờ.
  Nguy cơ cao hiện giới hạn dao/kéo/nĩa có bằng chứng phù hợp. Bút/đũa/đồng xu chưa
  có detector riêng; `other` vẫn là chưa xác định.
- Gate theo thời gian: ổn định mask, dwell, max-gap 250 ms, giữ bằng chứng tối đa
  800 ms. Ảnh cũ không tạo bằng chứng mới; reset lịch sử khi đổi nguồn/kích thước
  hoặc vòng xử lý quá 250 ms. Chưa đo đủ nhanh trên A03 hay chưa thì không kết luận.
- Monitor tách nguy cơ, khả năng quan sát, kết nối. Dữ liệu tự hết hạn dù yêu cầu
  mạng đang chờ. Mất kết nối không tô xanh hoặc xóa nguy cơ gần nhất. Tắt tiếng/
  cooldown chỉ tác động thông báo; âm báo chịu quyền và cài đặt của Android.
- QR máy chủ và QR ghép đôi một lần/hết hạn hai phút; Android Keystore bảo vệ cấu
  hình. HTTPS kiểm tra hostname/pin chứng chỉ. AES-256-GCM bảo vệ sự kiện; khóa
  không gửi cho relay. Không truyền/lưu video hay âm thanh mặc định.
- Relay FastAPI/SQLite: quyền monitor/camera riêng, thu hồi camera, chống lặp,
  hết hạn, reset phiên, lịch sử mã hóa có giới hạn và báo mất đoạn dữ liệu.
  Giao diện Cài đặt có quản lý và ngắt ghép đôi camera; cần thử trên máy thật.
- `scripts/companion_webcam.py` nối pipeline Python hiện có vào cùng kênh sự kiện.
  Hàng đợi tối đa bốn cảnh báo và một trạng thái mới nhất; không chặn inference
  bằng mạng, không ghi hình. Webcam thật tới điện thoại chưa được xác minh.
- Nút thử dùng `kind=test`, không giả phát hiện AI hoặc đổi nguy cơ. Tự kiểm tra
  mô hình bằng tensor tổng hợp; JSON không chứa ảnh, token hoặc địa chỉ relay.
- Lỗi nạp mô hình/mở camera giải phóng tài nguyên ngay cả khi màn hình còn giữ
  kết nối tới dịch vụ. Kết quả nạp mô hình đến muộn không bật lại camera đã tắt.
  Máy theo dõi bỏ trạng thái đang chạy khi luồng xử lý kết thúc và cho thử bật
  lại; nguy cơ gần nhất được giữ với trạng thái mất kết nối.
- Trạng thái dữ liệu mới tự hết hạn độc lập với yêu cầu mạng đang chờ, gồm cả
  thông báo trạng thái Android. Lỗi xác thực được giữ rõ trên giao diện; không
  bị câu thông báo nhận dữ liệu thành công ghi đè.

Android là bản chuyển riêng, **chưa chứng minh tương đương toàn bộ Python**.
Chưa chuyển ROI retry, One Euro và mọi quy tắc phụ. Các kết quả phải báo riêng.

## Thử trên A03 trước

1. Cài APK thử, vào **Cài đặt → Kiểm tra khả năng chạy mô hình trên máy**.
2. Chọn **Chạy kiểm tra**, giữ ứng dụng mở. Không cần ghép đôi hay mở camera.
3. **Xuất số liệu JSON** để đối chiếu. Có thể chạy thêm trên POCO.

Đây là thời gian mô hình với đầu vào tổng hợp, không phải FPS camera/recall/độ
trễ nhận cảnh báo. Bản 320 px nhằm thử cấu hình nhẹ; vật xa có thể kém hơn 640 px.
Ngưỡng Android còn ghi chưa kiểm chứng, không tô xanh vì chỉ nạp được mô hình.

## Hai điện thoại cùng Wi-Fi

Bản thử dùng relay HTTPS trên máy tính cùng Wi-Fi. Máy tính phải còn bật và IP
còn đúng. Chưa triển khai relay Internet/FCM hoặc bảo đảm khi Android ngủ lâu.
Không mở cổng Internet hay tự sửa firewall.

```powershell
rtk proxy uv pip install --python .venv/Scripts/python.exe -r requirements-companion.txt
rtk proxy .venv/Scripts/python.exe scripts/companion_certificate.py --host 192.168.1.10
rtk proxy .venv/Scripts/python.exe scripts/companion_relay.py --host 192.168.1.10 --cert outputs/companion/tls/cert.pem --key outputs/companion/tls/key.pem
```

Thay IP ví dụ bằng IP Wi-Fi thực. Tạo lại chứng chỉ sẽ đổi fingerprint và cần
cấu hình lại. Không chia sẻ khóa riêng. Chỉ lấy mã QR từ máy relay của nhóm.

1. Mở `outputs/companion/tls/relay-setup.png` trên máy tính.
2. POCO: **Ghép đôi → Quét mã cấu hình máy chủ trên máy tính**, sau đó **Tạo mã QR
   ghép đôi camera**.
3. A03: **Ghép đôi → Máy camera: quét mã ghép đôi**, quét QR trên POCO trong hai
   phút, rồi **Mở chế độ máy camera → Bật máy camera**.
4. POCO: **Theo dõi → Bật theo dõi**, cho phép thông báo.
5. Bấm **Gửi tín hiệu thử đến máy theo dõi** trên A03. POCO phải ghi rõ tín hiệu
   thử. Đây chỉ kiểm tra đường truyền/âm báo, không chứng minh AI nhận diện đúng.

Mạng khách cô lập thiết bị hoặc firewall có thể chặn kết nối; không tắt toàn bộ
firewall. Chỉ thử người tham gia đồng ý và đạo cụ an toàn; không đưa vật nguy hiểm
vào miệng, không diễn lại nghẹn. Ghi nhãn thật độc lập với dự đoán.

## Webcam máy tính

Trong QR ghép đôi trên POCO có nút hiện mã để chép sang máy tính. Lưu mã riêng
vào `outputs/companion/pairing.txt` và chạy lần đầu:

```powershell
rtk proxy .venv/Scripts/python.exe scripts/companion_webcam.py --pairing-file outputs/companion/pairing.txt --camera-index 0
```

Lần sau bỏ `--pairing-file`. Credentials ở `outputs/companion/`, không commit hay
chia sẻ. Thêm `--preview` để xem tại máy tính; Q hoặc Ctrl+C dừng. CLI hỗ trợ
webcam cục bộ, không nhận camera URL có mật khẩu.

## Kiểm tra đã thực hiện

- 33 kiểm thử Python companion qua: gồm HTTPS thật trên loopback, pin sai,
  ghép đôi/gửi/giải mã, quyền, hết hạn, hàng đợi và forwarder. Không phải 33 cảnh thật.
- Bộ Python đầy đủ: 312 qua, 3 bỏ qua. Lõi Kotlin: 30 kiểm thử JVM qua; không phải
  các cảnh quay điện thoại hay kiểm thử toàn bộ giao diện Android.
- 18 ca vòng đời dịch vụ qua với Robolectric 4.14.1 trên API 28/35: lỗi mô hình,
  lỗi mở camera, dừng trong khi nạp, thử bật lại và giữ nguy cơ cũ khi mất kết nối.
  Thay thế mô hình/phần cứng/mạng bằng thành phần thử; chưa kiểm tra tốc độ máy
  thật hoặc khả năng nhận cảnh báo khi Android ngủ.
- 12 ca nhận cảnh báo trên API 28/35 qua với HTTP mô phỏng: thực thi đường giải
  mã tới NotificationManager; kiểm tra sự kiện nguy cơ mới, snapshot cũ, tín hiệu
  thử, ciphertext bị sửa, dữ liệu hết hạn và trạng thái hết hạn khi HTTP đang chờ.
  Kênh cảnh báo có cấu hình âm thanh/độ ưu tiên cao; chưa chứng minh máy thật phát
  được âm báo dưới các cài đặt âm lượng, tiết kiệm pin hay khóa màn hình cụ thể.
- APK debug test03 biên dịch được, chữ ký v2 và checksum trọng số đã xác minh. Android API 26+,
  đa ABI. Chưa xác nhận trên A03/POCO.
- Classifier 11 lớp tại `outputs/with_other/runs/efficientnet_b0_group_robust/best.pt`:
  sai lệch logits tối đa khoảng `3.1e-6`, ba đầu vào tổng hợp/CPU Windows.
- Detector ONNX 320: `(1,116,2100)` và `(1,32,80,80)`. So sánh toàn bộ đầu ra một
  tensor với PyTorch: sai lệch tuyệt đối tối đa khoảng `7.48e-4` và `9.78e-6`.
  Tự kiểm tra điện thoại dùng logits classifier và 64 giá trị mỗi đầu ra detector;
  không kiểm tra độ chính xác cảnh thật hay toàn bộ hậu xử lý.
- Chưa có FPS camera, độ trễ hai thiết bị, recall sự kiện hoặc cảnh báo nhầm/giờ
  đo trên điện thoại; chưa đo nóng máy/chạy lâu/khóa màn hình.

## Dựng lại

JDK 17, Gradle 8.11.1, AGP 8.9.2, Kotlin 2.1.20, SDK 35. SDK/trọng số/APK/khóa/
cấu hình được Git bỏ qua. Người dùng máy hiện tại đã đồng ý điều khoản SDK;
người dựng trên máy khác phải tự đồng ý trước khi tải.

```powershell
rtk proxy uv pip install --python .venv/Scripts/python.exe -r requirements-android-export.txt
rtk proxy .venv/Scripts/python.exe scripts/export_android_models.py --checkpoint outputs/with_other/runs/efficientnet_b0_group_robust/best.pt --detector models/yolo11n-seg.pt --detector-size 320 --face models/face_landmarker.task --hand models/hand_landmarker.task --out outputs/android-models-a03
rtk proxy .venv/Scripts/python.exe scripts/package_android_models.py --source outputs/android-models-a03
rtk proxy .venv/Scripts/python.exe scripts/android_toolchain.py :core:test :app:testDebugUnitTest :app:assembleDebug
```

## Việc còn lại

Chờ số liệu/thử thật A03/POCO; đối chiếu tiền xử lý Android/Python, tracker/mask
và cảnh thật; thử mất mạng, khóa màn hình, thu hồi quyền, camera bị chiếm, nhiệt.
Còn thiếu hạ tầng Internet/FCM nếu ra ngoài LAN. Relay dành cho thử có kiểm soát,
chưa có xác thực tài khoản/
quản trị cho dịch vụ công khai. Xóa cấu hình cục bộ không tự thu hồi camera.
