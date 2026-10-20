# Phạm vi sản phẩm cần bảo vệ

## Mục đích

ChildSafe là nguyên mẫu hỗ trợ người chăm sóc chú ý khi vật cầm tay tiến sát vùng
miệng trên ảnh. Hệ thống chạy trên máy local, không lưu video mặc định và theo dõi
riêng từng người. Chưa chứng minh hiệu quả cảnh báo khi sử dụng với trẻ.

## Khả năng hiện có

- Định vị người và các vật thuộc model YOLO có sẵn; mốc mặt/tay trên crop của từng người.
- Phân loại vật với checkpoint đã huấn luyện; bộ lớp dữ liệu không phải đánh giá
  an toàn theo tuổi, kích thước hoặc hoàn cảnh.
- DANGER chỉ cho nhóm dao/dĩa/kéo đủ bằng chứng vùng vật, chất lượng quan sát và thời gian.
- CAUTION dành cho bằng chứng nghi ngờ; không đồng nghĩa đã xác nhận tiếp xúc.
- Mất quan sát được báo riêng; không dùng mốc miệng cũ để xác nhận hành động mới.

## Giới hạn phải giữ trong demo và lời bảo vệ

- Khoảng cách 2D không phải centimet hoặc bằng chứng vật đã vào miệng.
- Chưa định vị riêng bút, đũa, đồng xu, pin cúc. Phân loại được tên không đồng nghĩa định vị được.
- Chưa nhận diện nghẹn, nuốt hoặc xác định trẻ đang an toàn.
- Quay lưng từ đầu và che toàn bộ tay/vật/miệng có thể không có bằng chứng để phát hiện.
- Tối đa bốn ID là giới hạn thiết kế, không phải kết quả hiệu năng bốn người đã đo thật.
- CPU/GPU và cỡ detector phải ghi theo model thực sự chạy; không dùng timing GPU để mô tả máy CPU.

## Ba phần đóng góp nhóm có thể giải thích

1. Đánh giá phân loại có kiểm tra rò rỉ theo ảnh nguồn; so sánh độ bền và vật ngoài miền.
2. Kết hợp vật, tay, miệng và thời gian, với lịch sử riêng theo người và trạng thái quan sát riêng.
3. Công cụ ghi nhãn, lưu provenance, replay và đối chứng đường cơ sở trên cùng dữ liệu.

Đây là các công việc trong dự án, chưa phải khẳng định đầu tiên trên thế giới.
Phần 3 mới là khả năng đánh giá: kết luận giảm báo nhầm hoặc tăng recall cần phiên thật.

## Quyết định bối cảnh

Nhóm cần chọn một bối cảnh thử cụ thể: một khu vực trong nhà, camera cố định,
độ phân giải thực tế, số người, ánh sáng và nhóm vật. Chọn dựa trên số liệu thử,
không chọn góc đẹp sau khi nhìn kết quả test. Không nhận đã khảo sát người chăm sóc
hoặc có chuyên gia xác nhận khi chưa thực hiện.
