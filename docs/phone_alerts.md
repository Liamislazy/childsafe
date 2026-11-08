# Cảnh báo NGUY HIỂM trên điện thoại (Render + Web Push)

Camera và AI vẫn chạy trên laptop. Khi mức chung của khung hình tăng lên **CHÚ Ý** hoặc
**NGUY HIỂM**, laptop gửi lên dịch vụ nhỏ trên Render đúng ba thứ: mức, lý do, giờ. **Không gửi
ảnh hay video.** Render đẩy thông báo xuống các điện thoại đã mở trang và cho phép thông báo
(Chrome trên Android).

Mỗi mức có thời gian nghỉ (`PhoneNotifier.REPEAT_S`): NGUY HIỂM tối đa một lần mỗi 20 s, CHÚ Ý
một lần mỗi 60 s và không gửi CHÚ Ý trong 60 s sau một NGUY HIỂM. Báo theo mức chung, không theo
từng người: trong một lần thử webcam, một người bị gán 5 số người trong 39 s và mỗi số mới từng
làm điện thoại báo lại.

Rung và chuông do Android quyết định theo cài đặt thông báo của Chrome cho trang này; trang web
không tự bật được. Nếu thông báo đến mà không rung: Cài đặt → Ứng dụng → Chrome → Thông báo →
mục `childsafe-alerts.onrender.com` → bật âm thanh, rung, hiện nổi (tên mục khác nhau giữa
HyperOS và One UI).

```
laptop (camera + AI) --chữ, có mã bí mật--> Render: services/notify --Web Push--> Chrome Android
```

## 1. Tạo khóa và mã (một lần)

```powershell
uv venv .venv-notify --python 3.12
uv pip install --python .venv-notify\Scripts\python.exe -r services\notify\requirements.txt
.venv-notify\Scripts\python scripts\make_vapid_keys.py
```

Lệnh cuối in bốn dòng: `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `ALERT_TOKEN`, `MONITOR_CODE`.
Giữ riêng; không commit, không gửi vào nhóm chat. `MONITOR_CODE` là mã nhập trên điện thoại.

## 2. Tạo dịch vụ trên Render

1. Đẩy code lên GitHub (Render đọc `render.yaml` ở gốc repo).
2. Render → **New** → **Blueprint** → chọn repo `Liamislazy/childsafe` và nhánh có `render.yaml`.
3. Dán bốn giá trị ở bước 1 vào bốn biến môi trường Render hỏi, rồi **Apply**.
4. Khi deploy xong, mở `https://<tên>.onrender.com/api/health` phải thấy `{"ok":true,...}`.

Gói miễn phí: máy ngủ sau khoảng 15 phút không có truy cập và mất khoảng 30–60 s để thức dậy
(số liệu từ tài liệu bên thứ ba, chưa đối chiếu với trang chính thức của Render). Khi demo chạy,
laptop ping 10 phút một lần nên dịch vụ không ngủ, và tự khôi phục danh sách điện thoại nếu
dịch vụ khởi động lại (bản sao ở `outputs/phone_alerts/subscriptions.json`, không commit).

## 3. Điện thoại

Cách nhanh: khi demo đang chạy kèm cảnh báo (bước 4), bấm nút **Điện thoại** ở đầu trang demo.
Hộp thoại hiện mã QR chứa link trang cảnh báo và mã ghép đôi (sau dấu `#`, trình duyệt không gửi
phần này lên máy chủ). Quét bằng camera điện thoại → mở bằng **Chrome** → **Bật thông báo trên
điện thoại này** → cho phép. Hộp thoại tự đổi sang “Đã ghép 1 điện thoại ✓”. Laptop tự lấy mã
ghép đôi từ dịch vụ bằng `ALERT_TOKEN`, nên không cần nhập `MONITOR_CODE` ở đâu khác.

Cách thủ công: mở `https://<tên>.onrender.com` bằng Chrome, nhập `MONITOR_CODE`, bật thông báo,
**Gửi thông báo thử**. Mỗi điện thoại làm một lần.
Nếu bật tiết kiệm pin, Android có thể làm thông báo đến chậm. iPhone chỉ nhận khi đã “Thêm vào
Màn hình chính” (iOS 16.4+); chưa thử.

## 4. Chạy demo có gửi cảnh báo

```powershell
$env:CHILDSAFE_NOTIFY_TOKEN = "<ALERT_TOKEN>"
.venv\Scripts\python scripts\web_demo.py --accurate --notify-url https://<tên>.onrender.com
```

`/api/info` có mục `phone_alerts` (số đã gửi, số điện thoại, lỗi gần nhất). Nếu gửi lỗi, giao diện
ghi “Chưa gửi được cảnh báo tới điện thoại”. Demo vẫn chạy bình thường khi mạng lỗi.

## Kiểm thử

```powershell
.venv\Scripts\python -m pytest -q tests\test_phone_alerts.py
.venv-notify\Scripts\python -m pytest -q services\notify\test_app.py
```

Đã kiểm tra: mã hóa Web Push bằng khóa do script tạo được giải mã đúng ở phía nhận giả lập.
**Chưa kiểm tra** gửi qua dịch vụ push thật của Google tới điện thoại, độ trễ thực tế, hay
khi điện thoại khóa màn hình lâu.
