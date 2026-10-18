# Giao thức đặc trưng hình ảnh và âm thanh trên thiết bị

Ngày 12/10/2026. Module `childsafe.runtime.multimodal` là phần chuẩn bị đầu vào
cho nghiên cứu mô hình diễn biến theo thời gian. **Không có model nhận diện nghẹn,
không bật micro/camera và không đưa ra kết luận y khoa.** Runtime hiện tại chưa
được kết nối với module này. Các kiểm thử sử dụng số tổng hợp, không phải ca nghẹn.

## Producer và đồng hồ

- Mỗi đoạn đặc trưng gồm nguồn, generation kết nối, epoch phiên nghiên cứu,
  modality, khoảng capture `[start,end]`, subject và vector đặc trưng bất biến.
  Hình ảnh có thể dùng một timestamp; âm thanh phải có khoảng capture khác zero.
- Timestamp phải quy về một đồng hồ monotonic đã kiểm chứng. Cờ `clock_verified`
  mặc định false; không tự bật chỉ vì hai nguồn chạy trên cùng máy. Phải đo/ghi
  lại offset, độ không chắc chắn và độ trôi của timestamp capture trước khi bật.
  `FramePacket.received_t` hiện là lúc nhận ảnh, **không phải capture timestamp**;
  không dùng trực tiếp nó để tuyên bố đồng bộ hình ảnh với âm thanh.
- Callback âm thanh phải đưa metadata vào bộ đệm an toàn cho luồng; một consumer
  duy nhất gọi timeline. Module này không có khóa và không chạy trực tiếp trong
  callback của micro. Không ghi PCM hay frame vào log, không gọi dịch vụ đám mây.
- Định nghĩa/chuẩn hóa vector của từng producer phải có version riêng trước khi
  train; module chỉ kiểm tra số hữu hạn và tối đa 4096 chiều, không xác nhận ý nghĩa
  hoặc chất lượng của đặc trưng. Vector zero hợp lệ không phải nhãn nghẹn.

## Danh tính và kết nối

- `subject_id` phải là liên kết người đã xác minh trong phiên, không tự dùng cùng
  ID tracker ở hai camera. `attribution_verified` mặc định false. Không suy đoán
  trẻ dựa vào ngoại hình. Khi liên kết người không chắc chắn, không gán tín hiệu
  cho người đó; khi ID được dùng lại cho người mới, bắt đầu epoch mới.
- Âm thanh phòng chưa xác định người phát ra nằm trong `room_audio`. Nó không
  được đưa vào `audio` của bất kỳ người nào dù chỉ nhìn thấy một người. Người
  ngoài khung hình, TV hoặc người chăm sóc vẫn có thể là nguồn âm.
- Đăng ký nguồn trước khi append. Reconnect phải tăng generation; dữ liệu nguồn
  cũ bị xóa, callback đến muộn bị từ chối. Nếu mất đồng bộ đồng hồ hoặc danh tính
  liên camera, reset toàn bộ với epoch mới, duy nhất (không tái sử dụng epoch).
- Micro không hoạt động: `available=False, features=()`. Không thay bằng vector
  zero để giả vờ đã đo được sự im lặng. Cả clock chưa kiểm chứng lẫn nguồn lỗi đều
  làm cửa sổ không đạt điều kiện ghép tín hiệu.

## Cửa sổ thời gian và bộ nhớ

- Mặc định giữ tối đa 10 giây và 2048 bản ghi đặc trưng. Đây là lựa chọn kỹ thuật
  cho bộ đệm, **không phải ngưỡng phát hiện nghẹn đã hiệu chuẩn**. Nguồn đăng ký
  phải do ứng dụng kiểm soát; không dùng ID nguồn mới cho từng frame.
- `append(..., now=...)` và `window(..., now=...)` dùng cùng đồng hồ monotonic;
  now không được lùi. Gói tương lai, sai epoch/generation hoặc quá cũ bị từ chối.
- Query chỉ nhận đặc trưng có toàn bộ khoảng capture trong cửa sổ, không lấy
  thống kê âm thanh chứa tín hiệu sau mốc kết thúc hay trước mốc bắt đầu. Không
  nội suy qua khoảng trống và không dùng lại dấu người của phiên trước.
- `window.aligned` chỉ nghĩa là có các mẫu khả dụng, clock/subject được producer
  xác minh, khoảng trống không vượt dung sai cấu hình và cửa sổ chưa cũ. Nó không
  chứng minh quan sát đủ rõ để nhận diện nghẹn hoặc xác suất nghẹn thấp.
  Dung sai mặc định 0,25 giây cần chọn lại theo tần suất producer trên tập dev.
- Module không capture, không lưu file và không xuất chẩn đoán. Chưa có model
  đã kiểm chứng thì trạng thái sản phẩm vẫn là `choking assessment unavailable`.

## Kiểm chứng còn thiếu

Kiểm thử tự động bao phủ mất micro, âm thanh không rõ người, sai đồng hồ, đổi
người/phiên, reconnect, callback cũ, khoảng trống, dữ liệu tương lai, cửa sổ cũ,
giới hạn bộ đệm và chống sử dụng âm thanh ngoài cửa sổ. Cần đo đồng bộ capture
thật và trôi đồng hồ, xây producer đặc trưng, rồi train/đánh giá model riêng.
Chưa đo trên camera hoặc micro thật trong đợt này. Chưa có chuyên gia hay nhãn
lâm sàng; kết quả diễn lại không thay thế kiểm chứng ở trẻ.
