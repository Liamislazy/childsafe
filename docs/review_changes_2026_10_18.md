# Cải thiện sau rà soát mã nguồn — 18/10/2026

## 1. Danh tính và lịch sử

Tracker kiểm tra tốc độ dịch chuyển tương đối theo kích thước hộp người và thay đổi
tỷ lệ trước khi ghép ID. Phục hồi IoU thấp giữ số hiển thị nhưng reset lịch sử nguy
cơ, hình học, chất lượng và phân loại; người thay thế ở vị trí khác nhận ID mới.
Các ngưỡng là giá trị khởi đầu chưa hiệu chuẩn. Người khác xuất hiện đúng cùng vị
trí vẫn không phân biệt chắc chắn bằng tracker hình học; chưa có nhận dạng danh tính.

Kiểm chứng: các test tracking/identity/pipeline và hồi quy đổi người, đổi kích thước,
timestamp không tăng, phục hồi IoU thấp. Không có thử camera người thật ở bước này.
