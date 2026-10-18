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

## 3. Nguy cơ theo tình huống

Giữ nguyên checkpoint và nhãn lịch sử của dataset. Runtime hiện đại dùng chính
sách riêng: đồ chơi/trái cây/thìa gần miệng đủ bằng chứng yêu cầu chú ý ngay sau
cửa sổ xác nhận, không chờ 3 giây vì nhãn `safe`. DANGER hiện chỉ áp dụng nhóm
vật sắc/nhọn đã liệt kê, đủ xác suất, mask, quan sát và thời gian. Ổ điện/máy sưởi
cần ngữ cảnh khác; không tự gán nguy cơ nghẹn từ tên vật hay kích thước ảnh.
CSV legacy vẫn giữ luật lịch sử, được ghi rõ phiên bản chính sách riêng.

Kiểm chứng: nhóm safe, ngoài phạm vi, unknown, mask yếu, replay và hồi quy cảnh báo.
Chính sách là định nghĩa thử nghiệm, chưa có xác nhận y khoa hoặc đo hiệu quả thật.
