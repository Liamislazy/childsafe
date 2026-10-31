# Một phút chuẩn bị góc chơi cho con — bản thử

Mục tiêu: hỗ trợ người lớn rà soát một bàn/khay cố định trước khi trẻ tiếp cận.
Chỉ gợi ý các vật giống dao, kéo, nĩa bằng detector YOLO11n-seg cục bộ. Đây là
chế độ riêng với theo dõi vật gần miệng: không cần phát hiện người, mặt hoặc tay,
không dùng classifier ChildSUn để đoán một vật nằm ở đâu.

## Chạy

Trong môi trường runtime của dự án, có sẵn `models/yolo11n-seg.pt`:

```powershell
rtk proxy .venv/Scripts/python.exe scripts/play_area_demo.py --port 8002
```

Mở `http://127.0.0.1:8002/play-area`. CLI chỉ nghe localhost; không tải ngầm model.
`--weights` nhận một checkpoint detector có sẵn trên máy. Model không có đủ ba
lớp sẽ không được dùng để kết luận vật không còn xuất hiện.

Chế độ cũng có ở `/play-area` của `scripts/web_demo.py`, qua nút **Chuẩn bị góc
chơi**. Khởi động lại server cũ để nạp route mới. Bản chạy riêng nhẹ hơn vì không
nạp classifier hoặc MediaPipe. Không cần Android hay relay.

## Thử trong một phút

1. Dùng bàn có ánh sáng đều, mặt lót có chi tiết để kiểm tra độ nét. Camera nhìn
   xuống và giữ cố định. Chọn đúng camera nếu máy có nhiều webcam.
2. Căn phần bàn cần xem vào khung; thanh **Viền vùng rà soát** giới hạn vùng bên
   trong. Chỉ vùng này được phân tích. Đổi camera/vị trí bàn thì chọn **Phiên mới**.
3. Bấm **Bật camera**. Chế độ tự động lấy một ảnh mới sau khi ảnh trước đã xong;
   không tích lũy video. Có thể tắt tự động rồi bấm **Chụp ảnh kiểm tra**.
4. Chỉ thử với đạo cụ mô phỏng an toàn, không cần trẻ tham gia, không đưa vật vào
   miệng. Khi có vật nghi ngờ, đọc mục tương ứng bên phải và đối chiếu thực tế.
5. Cất đạo cụ và chờ hai lần cập nhật đủ rõ. UI chỉ nói **không còn phát hiện**;
   nguyên nhân vẫn có thể là che khuất hoặc nhận diện bỏ sót. Nếu tick **đã xử lý**
   khi máy còn thấy vật thì mục vẫn được nhắc, không tự hạ thành an toàn.
6. Tự kiểm tra các mục ngoài phạm vi. Bấm **Tải số liệu phiên** nếu muốn giữ JSON.

**Chọn ảnh bàn** là cách thử ảnh tĩnh, ghi rõ không phải cảnh hiện tại. Có thể
bấm chụp/kiểm tra lại sau khi chỉnh viền. Không trộn ảnh này với kết quả camera.

## Ý nghĩa của kết quả

- Mỗi mục là một ứng viên từ detector, không phải kết luận mức nguy hiểm hoặc
  vật đã nằm trong tầm với của trẻ. Điểm detector không phải xác suất gây tai nạn.
- Ảnh tối, lóa, thiếu chi tiết, kết quả cũ hoặc model lỗi không cung cấp bằng
  chứng để xóa mục trước. Các ngưỡng ảnh là giá trị khởi đầu, chưa hiệu chuẩn.
  Ảnh kém chất lượng vẫn có thể gợi ý vật cần kiểm tra nhưng không chứng minh
  vật đã biến mất; giao diện giữ trạng thái quan sát chưa đủ rõ.
- Việc gán cùng một vật qua ảnh dựa vào lớp và hộp ảnh ở góc cố định. Khi việc gán
  mơ hồ, không chuyển xác nhận của người dùng sang vật khác. Không tái nhận dạng.
- Bút, pin, đồng xu, ổ điện, máy sưởi, đồ chơi khác và kích thước vật không thuộc
  phạm vi đánh giá tự động. Đồ vật ngoài ba lớp không được xem là an toàn.
- Không lưu/gửi ảnh lên đám mây; dữ liệu ở RAM/tab. JSON do người dùng tải chỉ
  chứa tọa độ, trạng thái, model/checksum và các xác nhận do người dùng tự báo.
  Không phải nhãn ground truth và không tự chứng minh tai nạn đã được phòng ngừa.
- Lịch sử tối đa 80 mục, báo cáo giữ 20 lần quét gần nhất. Tạo phiên mới hoặc tải
  lại trang sẽ xóa lịch sử. Model chưa được đánh giá đầy đủ trên góc nhìn bàn.

## Kiểm chứng và phần còn thiếu

Kiểm thử API dùng detector giả để kiểm tra tọa độ, ảnh lỗi, model lỗi, bảo vệ
request và giới hạn dữ liệu. Kiểm thử Edge dùng camera/response mô phỏng để kiểm
tra tick không che vật đang thấy, hai lần không thấy, reset khi request đang
chạy, ảnh chưa rõ, xuất JSON và bố cục hẹp. Không mở camera vật lý trong các test.

Chưa có đánh giá bàn thật: cần cảnh không có người, bố trí đạo cụ được ghi nhận
độc lập, đo theo từng lớp/ánh sáng/kích thước ảnh/che khuất; báo precision, recall,
gợi ý sai mỗi bàn và thời gian xử lý. Không lấy ảnh huấn luyện hoặc demo chọn
sẵn làm bằng chứng hiệu quả tổng quát.

Muốn đánh giá giá trị với người chăm sóc: so sánh checklist thường với cùng
checklist được AI hỗ trợ trên các cảnh tương đương, cân bằng thứ tự và thống
nhất quy trình với trường/GVHD. Đo phần nguy cơ còn sót và hành động phù hợp,
không chỉ số lượt tick. Chưa có kết quả thì chưa ghi là giảm tai nạn hay giảm lo âu.

```powershell
rtk proxy .venv/Scripts/python.exe -m pytest -q tests/test_play_area.py
rtk proxy node tests/play_area_history.cjs
rtk proxy node tests/browser_play_area.cjs
```

Browser test cần Playwright + Edge; có thể đặt `CHILDSAFE_PLAYWRIGHT_MODULE` tới
gói Playwright đã cài. Ảnh chụp test UI nằm trong `outputs/play-area/`; ảnh có mục
kiểm tra là dữ liệu mô phỏng giao diện, không phải bằng chứng detector phát hiện.
