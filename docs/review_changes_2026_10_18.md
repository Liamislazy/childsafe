# Cải thiện sau rà soát mã nguồn — 18/10/2026

## 1. Danh tính và lịch sử

Tracker kiểm tra tốc độ dịch chuyển tương đối theo kích thước hộp người và thay đổi
tỷ lệ trước khi ghép ID. Phục hồi IoU thấp giữ số hiển thị nhưng reset lịch sử nguy
cơ, hình học, chất lượng và phân loại; người thay thế ở vị trí khác nhận ID mới.
Các ngưỡng là giá trị khởi đầu chưa hiệu chuẩn. Người khác xuất hiện đúng cùng vị
trí vẫn không phân biệt chắc chắn bằng tracker hình học; chưa có nhận dạng danh tính.

Kiểm chứng: các test tracking/identity/pipeline và hồi quy đổi người, đổi kích thước,
timestamp không tăng, phục hồi IoU thấp. Không có thử camera người thật ở bước này.

## 2. Đóng phiên camera

Hủy phiên và công bố kết quả dùng chung khóa ngắn. Xử lý model đang chạy được phép
kết thúc nhưng không ghi CSV hoặc giữ lịch sử nguy cơ sau hủy; thông tin phiên giữ
nguyên epoch đã xuất. Bắt đầu phiên giữ khóa xử lý trong suốt reset. Frontend ràng
buộc các tác vụ bất đồng bộ với thế hệ phiên để phản hồi cũ không tắt phiên mới.

Kiểm chứng: hồi quy dừng trong lúc giải mã và inference, quyền sở hữu/expiry,
reset khi bận, web runtime và nhãn quan sát. Chưa thử ngắt camera vật lý.
