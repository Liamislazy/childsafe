# Đồng bộ các commit main bị thiếu

Ngày 08/10/2026, nhánh `xkhoa/elevated-camera-implementation` trước đồng bộ là `b5e43d0`, tạo từ `6ddb27d`. Sau fetch, `origin/main` là `fd42b26`, có ba commit chưa được tích hợp:

- `6404a8e`: web demo so sánh classifier.
- `f19dbf5`: Open Images OOD và lớp `other`.
- `fd42b26`: nhiều lớp Open Images và class-balanced sampling.

Merge `origin/main` vào nhánh đã push để giữ toàn bộ lịch sử, không rebase/force-push và không sửa checkout gốc đang có thay đổi chưa commit. Không merge nhánh runtime vào main.

## Giải quyết xung đột và tương thích

- `.gitignore`: giữ cả data/external, dependency cục bộ và Ultralytics settings.
- Dataset/evaluation: giữ source roots nhiều nguồn; chấp nhận manifest bổ sung có đường dẫn tuyệt đối/source trống; cache phân biệt cả ảnh nguồn và preprocessing, giữ letterbox.
- Training: giữ kiểm tra group leak và các tập calibration riêng từ workflow bổ sung, đồng thời giữ WeightedRandomSampler sqrt của main. Không train hoặc tải dữ liệu mới trong lượt đồng bộ.
- Classifier: trả cả `top3` của web demo và `p_unknown_cal`; giữ preprocessing ghi trong checkpoint.
- Grad-CAM dùng preprocessing của checkpoint cho cả input và ảnh overlay, gồm letterbox.
- OOD và rejection metrics: áp dụng category/binary confidence và từ chối other/unknown/empty_hand nhất quán với runtime. Các kết quả số cũ trên main vẫn là báo cáo lịch sử với logic thời điểm đó; không tái chạy hay sửa số kết quả cũ.
- Web demo vẫn là comparator classifier một người, không phải runtime segmentation nhiều người. Không có mask thì không xác nhận danger/hiển thị xanh; trả observation và lý do hạn chế. Photo upload không gọi `other` là safe; `top3` vẫn được giữ.
- Tái tạo cả notebook từ source đã tích hợp. Giữ nguyên các manifest split và báo cáo đã được commit trên main; không thêm ảnh, trọng số hoặc video.

## Kiểm thử

Thêm hồi quy cho nhiều source có cùng tên ảnh, manifest bổ sung tuyệt đối/source thiếu, cache letterbox, sampling ưu tiên lớp ít mẫu, top3 + unknown probability, OOD/rejection không coi other là safe, và endpoint web qua FastAPI TestClient (không mở camera).

Các giới hạn nghiệm thu webcam/calibration/GPU và detector pen/coin trong `runtime_validation.md` vẫn áp dụng. Không chạy training hoặc tải dataset.

Suite đầy đủ: 70 kiểm thử đều qua, gồm integration với model thật và overlay Grad-CAM letterbox. Ba cảnh báo deprecation từ dependency; không có kiểm thử bị bỏ qua. Compileall source, scripts và tests cũng qua.
