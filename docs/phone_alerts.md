# Cảnh báo NGUY HIỂM trên điện thoại (Render + Web Push)

Camera và AI vẫn chạy trên laptop. Khi một sự cố mới lên **NGUY HIỂM**, laptop gửi lên dịch vụ
nhỏ trên Render đúng ba thứ: mức, lý do, giờ. **Không gửi ảnh hay video.** Render đẩy thông báo
xuống các điện thoại đã mở trang và cho phép thông báo (Chrome trên Android). CHÚ Ý không bắn
thông báo; thông báo lặp bị chặn trong `temporal.cooldown_s` (5 s) sau khi NGUY HIỂM kết thúc.

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

Mở `https://<tên>.onrender.com` bằng **Chrome trên Android** → nhập `MONITOR_CODE` →
**Bật thông báo trên điện thoại này** → cho phép → **Gửi thông báo thử**. Mỗi điện thoại làm một lần.
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
