# Vật nhỏ ở xa: sửa runtime và giới hạn còn lại (21/10/2026)

Người thử báo bút cách camera khoảng 2 m. YOLO11n-seg đang dùng không có lớp bút;
classifier 11 lớp cũng không có bút. Một vùng bị gọi là toothbrush/cell phone
không chứng minh nhận diện bút đúng. Không bổ sung một tên lớp trong cấu hình
để coi như đã có trọng số được train.

## Thay đổi

- Khi thấy mặt nhưng bỏ sót tay, thử MediaPipe hand-only trên một vùng ảnh gốc
  quanh miệng, tối đa một người mỗi frame và luân phiên các ID. Giữ phép đổi
  tọa độ về ảnh gốc, điều kiện gán tay theo người và các ngưỡng chất lượng gốc.
  Không thử khi mặt quá nhỏ, người giao nhau hoặc thiếu thời gian xử lý.
- Với hai tay cách xa nhau, dò vật theo từng tay cùng vùng miệng, luân phiên
  thay vì gộp cả hai thành một crop quá lớn rồi bỏ qua. Vẫn tối đa một crop
  detector phụ mỗi frame theo cấu hình mặc định.
- Khi detector có hộp vật nhưng mask rỗng, quá ít pixel, sai kích thước hoặc
  phân mảnh, giữ khoảng cách hộp tới vùng miệng làm bằng chứng ứng viên. Hộp
  chỉ tạo CAUTION sau thời gian tồn tại; không tạo mask giả hay xác nhận DANGER.
  Mask đáng tin cậy và ổn định theo thời gian vẫn là điều kiện của bằng chứng mạnh.
- Kiểm tra thời gian còn lại và chi phí đo được trước các lượt dò vật/phân loại
  phụ. Khi không đủ, giữ kết quả lượt chính và ghi `roi_budget_skips`; không cố
  chạy thêm rồi dùng ảnh quá hạn. Ước lượng này không bảo đảm thời gian thực.
- Tắt mặc định classifier trên crop tay khi chưa định vị được vật: kết quả này
  vốn chỉ dùng chẩn đoán, không được tạo cảnh báo, nhưng vẫn tốn thời gian CPU.
- CSV/API ghi nguồn vị trí ứng viên (mask/hộp), kích thước hộp/mask và lượt tìm
  tay bổ sung. Giao diện nêu rõ hộp chỉ đủ cảnh giác và khi bỏ qua lượt dò vì thiếu
  thời gian. Tay không cầm vật vẫn không được khởi tạo cảnh báo.

Giữ `max_gap_s=0.25`, `stale_frame_s=0.25`, giữ bằng chứng mất dấu tối đa 0,8 s,
không nới chất lượng ảnh, không đổi trọng số hoặc tuyên bố camera đã hiệu chuẩn.
Không tự thu ảnh/video; nút tải ảnh chỉ hoạt động khi người dùng chủ động bấm.

## Kiểm chứng đã làm

Bộ kiểm thử gồm ánh xạ crop tay về ảnh gốc, luân phiên bốn người, không tạo cảnh
báo từ tay trần, hai tay cách xa nhau, mask hỏng chỉ tạo CAUTION, khoảng cách từ
hộp vật dài tới vùng miệng và bỏ qua lượt dò phụ không kịp deadline. Các kiểm thử
cũ về mất dấu, replay, che khuất và báo nhầm khi sờ mặt vẫn được giữ.
Kết quả toàn bộ suite: **279 passed, 3 skipped**, 17 cảnh báo dependency. Đây là
kiểm chứng phần mềm, không phải kiểm chứng độ chính xác camera ở 2 m.

Đọc phiên đã kết thúc ngay trước sửa: 393 dòng feature, 291 dòng hợp lệ, 168 dòng
thấy miệng và 44 dòng có tay; chỉ một dòng đạt ổn định mask/bằng chứng khoảng cách
mạnh. Đây là số dòng chẩn đoán, không phải recall. Phiên không có nhãn sự kiện
độc lập nên không tính được tỷ lệ bỏ sót hay báo nhầm, và không xác định được
đoạn nào thực sự là người đứng ở 2 m.

Đã chạy model local trên ảnh chụp màn hình cũ của người thử, cắt vùng camera rồi
thu nhỏ trên nền xám. Ảnh có overlay cũ, không phải ảnh camera gốc. Với cả chế độ
cũ và mới, model gọi vật là toothbrush ở tỷ lệ 0,7, cell phone ở 0,5; không định
vị vật ở 0,35; không phát hiện người ở 0,25. Lượt tìm tay bổ sung không chạy vì
lượt chính đã thấy tay. Mỗi điều kiện chỉ có ba mẫu sau ba lượt khởi động; bật dò
bổ sung tăng thời gian xử lý trong thử nghiệm này. Kết quả không chứng minh cải
thiện nhận bút, FPS, khoảng cách 2 m hay độ chính xác sự kiện. Không quy đổi tỷ lệ
thu nhỏ ảnh thành khoảng cách vật lý.

## Việc cần làm trên camera thật

1. Bật camera, giữ ánh sáng và độ phân giải cố định. Thử lần lượt 1 m, 1,5 m và
   2 m với cùng đạo cụ lớn, tù; chỉ đưa sát phía ngoài vùng miệng, không cho vào
   miệng. Có cả cầm vật xa mặt và sờ má/gãi mặt không cầm vật làm đối chứng.
2. Ghi nhãn bắt đầu/kết thúc độc lập trước khi xem thống kê; tải feature, nhãn và
   JSON phiên trước khi reset. Khi cần chẩn đoán, chủ động bấm **Tải ảnh kiểm tra**
   ở từng khoảng cách để xem số pixel thật của tay, đầu bút và miệng.
3. Bút chưa được hỗ trợ: cần ảnh gốc gán hộp/mask bút, cả mẫu khó ở xa và mẫu âm
   tính sờ mặt; chia theo người/phiên trước train và kiểm tra bằng người/phiên
   chưa dùng để chọn ngưỡng. Workflow Kaggle và detector bổ sung đã có trong
   `expanded_hazard_runtime.md`; chưa có trọng số bút đã train để bật mặc định.
4. Báo riêng recall ở từng khoảng cách, báo nhầm/giờ trên đoạn âm có nhãn, độ trễ
   và tỷ lệ quan sát được. Giữ các đoạn mất quan sát trong mẫu số recall tổng.

Chưa nghiệm thu bút ở 2 m. Sửa runtime chỉ loại các điểm làm mất bằng chứng định
vị đã có; không thay thế detector chưa học lớp vật và chi tiết ảnh bị thiếu.
