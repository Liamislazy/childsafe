# Việc nhóm cần xem xét và làm thật

Các mục này không được phần mềm hoặc tài liệu mới tự giải quyết.

| Ưu tiên | Việc còn thiếu | Người phụ trách đề xuất | Minh chứng cần có |
|---|---|---|---|
| Trước thu dữ liệu | Xác nhận quy trình với giáo viên, người lớn đồng ý, đạo cụ và cách giữ dữ liệu | Khoa + người hướng dẫn | Xác nhận thực tế; không tự điền phê duyệt |
| Trước trình diễn | Chốt bối cảnh, máy, camera, độ phân giải, số người, nhóm vật | Cả hai | Kết quả thử trên đúng thiết bị, không chỉ tên cấu hình |
| Cao | Đo lại lỗi theo dõi/mặt/tay và cảnh báo sau các sửa code | Lâm chạy, Khoa ghi nhãn độc lập; đổi vai khi cần | CSV/JSON, nhãn, cấu hình/model/source hash |
| Cao | Thu test mới và nền không sự kiện; không chỉnh ngưỡng bằng test | Cả hai | Recall tổng, CAUTION riêng, FA/giờ, trễ, FPS, độ phủ và thất bại |
| Cao | Hiểu và kiểm tra phần mã AI hỗ trợ | Từng thành viên | Giải thích được mã, quyết định và ca lỗi; nhật ký đúng đóng góp |
| Cao | Chuẩn bị poster, demo và luyện phản biện | Lâm hình/bảng; cả hai trình bày | Tệp poster đúng mẫu, đường dẫn demo, minh chứng đọc được |
| Sau thử nhu cầu | Xác định ai nhận cảnh báo và phản hồi của người chăm sóc | Khoa + người hướng dẫn | Trao đổi/đánh giá theo quy trình trường cho phép; chưa có thì ghi là giả định |
| Phát triển dài hạn | Thu dữ liệu và định vị bút/đồng xu/pin cúc theo mục tiêu thật | Cả hai + người hướng dẫn | Nhãn vị trí và nhóm nguồn độc lập, model và kiểm thử riêng |
| Nghiên cứu khác | Nhận diện nghẹn bằng hình/âm thanh | Cần chuyên gia và dữ liệu xác nhận | Chưa có; không đưa thành tính năng đã hỗ trợ |

## Khi phải thu hẹp phạm vi

Nếu tốc độ hoặc khả năng thấy mặt/tay không đạt ở góc cao/hai người, nhóm cần tự
quyết định thu hẹp bối cảnh trình diễn theo số liệu và trình bày trung thực. Không
đổi trạng thái `calibrated`/`complete_object_coverage` để làm giao diện xanh.

## Chưa thể tuyên bố

- Giảm tai nạn, cứu trẻ hoặc ngăn nghẹn.
- Chạy tốt ở mọi góc hoặc xác nhận tiếp xúc khi miệng bị che.
- CPU rẻ chạy đủ nhanh trong điều kiện chưa đo.
- Phần mềm đã được bác sĩ, phụ huynh hoặc trường chấp thuận khi chưa có xác nhận.
- Sẵn sàng triển khai chỉ vì test tự động qua hoặc demo có một cảnh báo đúng.
