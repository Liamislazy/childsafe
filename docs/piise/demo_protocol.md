# Kịch bản trình diễn có kiểm soát

Chỉ người lớn đã đồng ý hoặc ma-nơ-canh. Đạo cụ mềm/đầu tù, lớn, không nuốt được;
dừng trước miệng. Xác nhận quy trình với giáo viên trước khi thu dữ liệu người.
Không dùng video trẻ hoặc gây/diễn bằng cách nín thở, nghẹn.

## Trước khi trình diễn

- Ghi máy, camera, độ phân giải ảnh trả về, tên/hash classifier và detector thực tế.
- Mở ứng dụng local, đọc phần phạm vi hỗ trợ. Camera chỉ bật khi người dùng chủ động chọn.
- Kiểm tra tốc độ, không tăng `max_gap_s` chỉ để làm cảnh báo xuất hiện.
- Phân công một người thao tác và một người ghi nhãn độc lập. Nhãn theo hành động,
  không theo màu cảnh báo. Nếu không biết tình huống, ghi uncertain.
- Lưu các tệp số trước khi reset. Nếu phiên dùng để đánh giá, chuẩn bị manifest trước.

## Trình tự ngắn dành cho giám khảo

| Cảnh | Hành động | Điều cần quan sát | Không hứa trước |
|---|---|---|---|
| Khởi động | Camera chưa bật | Không tự truy cập camera; xem model và phạm vi | Suy luận model đã kiểm chứng trên trẻ |
| Nền thường | Tay/vật ở xa vùng miệng | Mức nguy cơ và khả năng quan sát được hiển thị riêng | Mọi nhãn safe đều an toàn |
| Tiến gần | Đạo cụ thuộc loại đã hỗ trợ tiến gần, dừng trước miệng | Vùng vật, khoảng cách 2D, bằng chứng theo thời gian và cảnh báo nếu đủ điều kiện | Đạo cụ chắc chắn được model nhận ra |
| Nhiễu | Tay chạm má/cằm, không có vật mục tiêu | Báo riêng nghi ngờ và nguy cơ cao; ghi mọi báo nhầm | CAUTION là DANGER đúng |
| Che khuất | Mất mặt/tay/vật, rồi quan sát lại | Không tự suy ra an toàn hay tích lũy tiếp xúc từ mốc cũ | Quay lưng vẫn xác nhận được hành động bị che |
| Kết thúc | Đóng marker, tắt camera, xuất ba tệp số | Phiên kết thúc rõ; không trộn số liệu sau reconnect | Phiên demo đã là kiểm thử độc lập |

Không dựng cảnh hoặc đổi nhãn chỉ để demo có màu đỏ. Nếu vật không được tìm thấy,
trình bày đúng thất bại và phân biệt lỗi detector, mốc, classifier, tốc độ hoặc luật.

## Đánh giá khác với trình diễn

Thu dev để xem lỗi; chốt cấu hình rồi thu test mới bằng người/phiên độc lập theo
hướng dẫn runtime. Giữ cả sự kiện bị che/mất quan sát trong recall tổng. Có thời
gian nền được ghi rõ `non_event` để tính báo nhầm/giờ; thời gian uncertain không
được tính âm tính. CAUTION và DANGER có bảng riêng. So sánh đường cơ sở theo
cùng sự kiện; muốn so ở cùng recall phải chọn ngưỡng trên dev trước.

Ma-nơ-canh hoặc đạo cụ có thể không giống ảnh model đã học. Kết quả với đạo cụ
chỉ mô tả đạo cụ và người lớn trong phiên đó; không chuyển thành hiệu quả ở trẻ.
