# Train detector bút và đũa

## Vì sao checkpoint classifier cũ chưa nhận ra vị trí bút?

`fetch_openimages.py` cắt ảnh vật về 224×224 để train phân loại. `Pen` được gom vào `other`; `Chopsticks` không nằm trong mapping của script đó. EfficientNet nhận một ảnh/crop rồi trả xác suất lớp, không trả hộp vị trí hay mask. Detector camera `yolo11n-seg.pt` hiện tại là checkpoint COCO, chưa được fine-tune trên bút/đũa.

Luồng mới dùng ảnh nguyên khung và bounding box Open Images: hai lớp riêng `pen`, `chopsticks`. Model YOLO11n detection nhận khung camera và trả vị trí/lớp/confidence. Bounding box chưa phải mask và không xác định chính xác đầu vật; không chuyển hộp chữ nhật thành mask giả để xác nhận danger.

## Dữ liệu đã chuẩn bị

Gói ban đầu: 300 ảnh train / 44 validation / 84 test; tải thành công toàn bộ 428 ảnh. Giữ split chính thức, kiểm tra ImageID và SHA-256 không trùng giữa split. Mỗi ảnh giữ tất cả bbox của hai lớp, bỏ ảnh có group/depiction, bỏ ảnh bị xoay. Attribution tác giả/license/URL trong `manifest.csv`.

```powershell
.venv\Scripts\python scripts\prepare_small_object_detector.py --train-cap 150 --eval-cap 50 --workers 8
.venv\Scripts\python scripts\train_small_object_detector.py --epochs 10 --imgsz 384 --batch 8 --device cpu
```

Đây là baseline nhỏ. Chưa có tập cảnh camera cầm bút/đũa được gán nhãn hoặc đủ ảnh nền âm tính để nghiệm thu báo nhầm. Tăng train-cap ở lượt dữ liệu tiếp theo và bổ sung dữ liệu thực tế; giữ test độc lập. Không đưa ảnh kiểm tra của người dùng vào train tự động.

Lượt CPU ngày 08/10/2026 đã hoàn tất 10 epochs, imgsz 384, khoảng 389 giây train. Checkpoint: `outputs/pen_chopsticks_detector_cpu/weights/best.pt`. Test riêng: precision 61,04%, recall 46,68%, mAP50 53,91%, mAP50–95 30,45%. Thử ảnh webcam bút đã cung cấp, chuyển về 640×480, ngưỡng confidence 0,25: không có detection. Ảnh đó không được dùng để train. Kết quả này chưa đủ để thay luồng camera; model CPU chưa được bật trong app. Báo cáo tại `evaluation.json` và `provided_image_check.json` trong thư mục kết quả.

Mỗi lượt chuẩn bị hoặc train mới phải dùng thư mục `--out` / `--output` mới để giữ dữ liệu và kết quả cũ; script không ghi đè bộ đã hoàn tất.

## Kaggle

1. Upload `childsafe_pen_chopsticks_detection.zip` thành Kaggle Dataset riêng. ZIP có images/labels/manifest/report và backbone local.
2. Import `notebooks/childsafe_detector_kaggle.ipynb`, thêm dataset vừa upload.
3. Bật GPU và Internet, Run All. Notebook train 50 epochs, ảnh 640, batch 16; nếu GPU hết bộ nhớ, giảm batch xuống 8.
4. Lấy `pen_chopsticks_detector_results.zip` trong Output. Bên trong có `weights/best.pt`, `results.csv`, và `evaluation.json` cho test riêng.

Notebook tìm dataset qua manifest và viết lại đường dẫn Kaggle. Nó chuyển backbone segmentation COCO đang có sang kiến trúc detection, rồi train đầu ra hai lớp. Không thay thế checkpoint phân loại 11 lớp.

## Đánh giá và đưa vào camera

So sánh precision/recall/mAP theo từng lớp trên test, sau đó kiểm tra video độc lập ở độ phân giải webcam thật: cầm ngang/dọc, che bởi tay, đầu vật sát miệng, không có vật, hai người và ánh sáng kém. mAP ảnh tĩnh không thay cho event recall, báo nhầm/giờ và độ trễ.

Không tự thay detector người bằng model hai lớp: model này không có lớp person. Khi đủ kiểm chứng, ghép detector người hiện tại và detector vật riêng, kiểm tra liên kết vật–tay–người. Hộp bút/đũa có thể tạo cảnh giác nghi ngờ; muốn xác nhận đầu vật sát miệng cần nhãn segmentation/keypoint và lượt train tiếp theo. Giữ `complete_object_coverage=false`.

Nguồn: [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html), [Ultralytics detection](https://docs.ultralytics.com/tasks/detect), [segmentation](https://docs.ultralytics.com/tasks/segment).
