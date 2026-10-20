# Chuẩn bị trình diễn và minh chứng PIISE

Bộ bổ sung ngày 21/10/2026. Không thay thế hoặc chỉnh sửa báo cáo đã chốt trên Drive.
Các nội dung ở đây giúp nhóm rà soát, thử nghiệm và tự chuẩn bị phần bảo vệ theo
đúng quy định về AI của trường. Không có kết quả camera hoặc dữ liệu người mới.

## Bắt đầu

1. Đọc [phạm vi sản phẩm](product_scope.md) và chọn bối cảnh nhóm thực sự có thể kiểm tra.
2. Kiểm tra model local, chưa mở camera:

   ```powershell
   .venv\Scripts\python scripts/check_demo_readiness.py
   ```

3. Dùng [kịch bản trình diễn](demo_protocol.md), sau khi thống nhất quy trình với giáo viên.
4. Tạo/đóng phiên theo [hướng dẫn runtime](../product_runtime_guide.md) và
   [hướng dẫn nhãn độc lập](../targeted_camera_trials.md). Không tự sửa config/hash của phiên đã thu.
5. Xuất bảng đối chứng từ phiên test mới:

   ```powershell
   .venv\Scripts\python scripts/prepare_fair_evidence.py --sessions outputs/scenarios --split test --out outputs/piise_evidence
   ```

   Không có phiên: CSV chỉ có tiêu đề; kết quả ghi chưa đo. Phiên hỏng/chưa hoàn tất:
   công cụ dừng, không tạo số thay thế. Có phiên: kiểm tra provenance, dùng cùng bộ
   metric và xuất `summary.md`, `comparison.csv`, `evidence.json`. Số liệu gốc vẫn ở thư mục phiên.

6. Dùng [câu hỏi phản biện](defense_questions.md) và [nội dung dành cho poster](poster_brief.md)
   để tự chuẩn bị, không sao chép thành báo cáo mới do AI viết.
7. Xem [việc cần nhóm quyết định và làm thật](remaining_work.md).

## Phân biệt ba loại bằng chứng

| Bằng chứng | Cho biết | Không cho biết |
|---|---|---|
| Test tự động với dữ liệu giả | Quy tắc, phiên, tọa độ, replay có hành vi như dự kiến | Camera nhìn thấy trẻ hoặc nhận diện đúng sự kiện |
| Model thật trên ảnh vật/ảnh hình học | Model local nạp và suy luận được, thời gian trong workload đã đo | Recall sự kiện, báo nhầm/giờ, độ trễ sử dụng thực tế |
| Phiên có nhãn độc lập | Chất lượng cảnh báo trong chính các điều kiện đã thử | Hiệu quả ở trẻ hoặc ở mọi cách bố trí camera |

`comparison.csv` tách DANGER, CAUTION-or-DANGER và đường cơ sở khoảng cách. Không
chọn ngưỡng bằng test và không bỏ các sự kiện mất quan sát khỏi recall tổng.
Kiểm thử cùng một người nhiều lần không tương đương nhiều người độc lập.
