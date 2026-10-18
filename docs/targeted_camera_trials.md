# Thu phiên thử tập trung vào lỗi camera

Chỉ người lớn đồng ý tham gia hoặc ma-nơ-canh. Dùng đạo cụ mềm, không sắc, mô phỏng
chi tiết nhỏ bằng vật lớn không nuốt được; dừng trước miệng, không đưa vật vào miệng,
không gây hoặc diễn bằng cách nín thở/nghẹn. Nhãn là tình huống vật tiến sát vùng
miệng trên ảnh hoặc mốc hành động người quan sát thấy, không phải nghẹn lâm sàng.

Tạo checklist:

```powershell
.venv\Scripts\python scripts/plan_camera_trials.py --repeats 1 --out outputs/planned_trials.csv
```

Mỗi split có 20 điều kiện trọng tâm, gồm vật dài, ảnh nhỏ/thiếu sáng, quay lưng,
che khuất ngắn/dài, đổi người, giao nhau, thao tác thường ngày và stress bốn người.
Đây là kế hoạch chưa đo; cột kết quả để trống. Phân người/buổi quay độc lập cho
dev/test trước khi chạy; không đổi ngưỡng theo test. Mỗi điều kiện cần cả đoạn
`non_event` rõ nhãn và sự kiện mô phỏng thích hợp; dùng `uncertain` khi người gán
nhãn cũng không chắc. Ghi thời gian thực tế, không điền dự kiến thành kết quả.

Tạo manifest bằng `scenario_session.py` theo hướng dẫn sản phẩm. Cấu hình/model
phải khớp bản web đang chạy; truyền `--config` và `--model` tường minh khi cần.
Một người điều khiển đạo cụ, một người quan sát riêng đánh dấu bắt đầu/kết thúc.
Đánh dấu dựa vào hành động, không nhìn màu cảnh báo để quyết định nhãn. Không
sửa nhãn để làm metric đẹp. Ghi độ trễ thao tác người gán nhãn là chưa đo nếu thiếu.

Trước khi tắt camera, kết thúc marker đang mở. Sau khi tắt, tải đủ ba file
`camera_session.csv`, `annotations.csv`, `session.json` rồi đóng phiên đã chuẩn bị:

```powershell
.venv\Scripts\python scripts/finalize_session.py outputs/scenarios/dev01 --features outputs/browser/camera_session.csv --annotations outputs/browser/annotations.csv --metadata outputs/browser/session.json
```

Công cụ kiểm tra hash model/config/source, epoch và khoảng thời gian; từ chối
phiên có marker chưa đóng hoặc dữ liệu sau lúc kết thúc. Không ghi video. Phiên dev
có template nhãn khả năng thấy miệng để điền độc lập; bỏ dòng chưa gán nhãn trước
khi đề xuất ngưỡng, luôn giữ cả ví dụ thấy rõ và không thấy rõ. CSV tọa độ không
cho phép người gán nhãn xem lại ảnh: nếu không quan sát trực tiếp được thì để nhãn
trống, không suy từ `mouth_visible` của hệ thống để tạo ground truth.

Chốt cấu hình trên dev, thu test mới bằng cấu hình đó, rồi báo cáo recall tổng,
FA/giờ âm tính rõ nhãn, latency, image coverage và strict coverage theo điều kiện.
Thời gian mất quan sát vẫn nằm trong mẫu số đầy đủ. Chưa có dữ liệu thử hợp lệ thì
các mục trên là chưa kiểm chứng; không chuyển kết quả người lớn sang hiệu quả ở trẻ.
