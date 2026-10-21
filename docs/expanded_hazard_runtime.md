# Sửa lỗi thử camera và mở rộng vật nguy cơ

## Đã thay đổi gì

- Detector và classifier có dự đoán loại riêng. Classifier 11 lớp trả `other` không còn phủ nhận
  bằng chứng detector đủ mạnh về dao/dĩa/kéo. Điểm detector tối thiểu 0,75 là cấu hình ban đầu
  **chưa hiệu chuẩn**, không phải xác suất vật nguy hiểm.
- DANGER vẫn cần miệng quan sát được, vật liên kết đúng tay/người, mask đủ rõ và ổn định,
  khoảng cách 2D cùng thời gian duy trì. Hai model nhận khác nhau chắc chắn chỉ tạo CAUTION.
  Bounding box, tay gần miệng hoặc vật lạ không tự trở thành mask xác nhận.
- Web/CSV ghi nguồn đánh giá loại nguy cơ. Thẻ không có nguy cơ nhưng quan sát hạn chế dùng nền xám.
- Cache phân loại tối đa 0,3 giây khi cùng vật; đổi vật xóa cache. Giữ ảnh gốc cho bộ mốc.
  Giữ mặc định luồng thư viện vì ép 2 luồng chậm hơn trong lượt đo này; CLI cho phép chọn riêng.
  Bỏ chờ vẽ giao diện dư thừa sau mỗi kết quả.
  Không tăng `max_gap_s`, không kéo dài tuổi frame để ép DANGER xuất hiện.
- Có thể ghép detector vật riêng qua `--object-model`. Detector người hiện tại vẫn được giữ;
  model bổ sung chạy trên crop tay–miệng, tối đa một ROI mỗi frame theo cấu hình mặc định.
  Model bổ sung thiếu/lỗi làm chế độ hạn chế hiện rõ. Hash phiên tính cả hai trọng số.

## Các lớp thực tế và các lớp cần train

Trong máy hiện có detector COCO YOLO11n-seg và classifier 11 lớp. Ba loại cảnh báo cao có sẵn
là `knife`, `fork`, `scissors`; chưa có trọng số đạt yêu cầu cho bút hoặc đồng xu.

`configs/hazard_classes.yaml` định nghĩa **15 mục tiêu gán nhãn**:

| Nhóm | Lớp |
|---|---|
| Có trong detector hiện tại | dao, dĩa, kéo |
| Cần dữ liệu/trọng số bổ sung | bút, bút chì, đũa, đồng xu, pin cúc, pin trụ, ốc vít, đinh, kim băng, kim, viên bi, chi tiết đồ chơi nhỏ |

Đây là danh mục train, **không phải kết quả nhận diện 15 lớp**. Giao diện chỉ liệt kê lớp bổ sung
đang có trong model thực sự nạp được. Có tên lớp cũng không chứng minh nhận tốt ở xa.
Không đổi `complete_object_coverage` hoặc `quality.calibrated` thành true chỉ để đổi màu giao diện.

Starter hiện có trong `data/supplement` là crop phân loại: 29 bút, 23 đồng xu, 56 dao,
25 dĩa, 1 kéo và 35 thìa đã qua lượt review trước. Chúng chưa phải bộ full-frame segmentation
và không đủ cơ sở tuyên bố đã học tốt vật nhỏ ở camera góc phòng. Không tạo mask từ các crop/bbox này.

## Chuẩn bị dữ liệu có nhãn thật

1. Ưu tiên bút/đũa/kéo trước nếu thời gian hạn chế. Chụp ảnh nguyên khung, có cảnh gần/vừa/xa,
   hướng cầm khác nhau, một phần bị tay che, nền tương tự phòng thử; thêm bàn tay trống,
   gãi mặt/chống cằm, điện thoại, bàn chải để kiểm tra nhận nhầm. Thu người lớn đồng ý tham gia
   hoặc đạo cụ; không đưa kim, pin, đồng xu hay đồ sắc tới miệng để tạo dữ liệu.
2. Gán nhãn thủ công phần vật nhìn thấy. Segmentation cần polygon, không dùng hình chữ nhật
   làm mask giả. Nếu chỉ có bbox, chọn `--task detect`; camera chỉ dùng nó làm cảnh giác.
3. Lưu mỗi ảnh cạnh một `.txt` YOLO. ID lớp ban đầu lấy từ `configs/hazard_classes.yaml`.
   Ví dụ bút có ID 3. Ảnh âm tính có `.txt` rỗng sau khi người gán nhãn đã kiểm tra toàn ảnh.
4. Tạo CSV theo `docs/hazard_annotations_template.csv`, một dòng mỗi ảnh. `reviewed` và
   `annotation_complete` phải là true sau review; `source`, `license` ghi nguồn/quyền sử dụng thực tế.
   Ảnh công khai cần giữ attribution tác giả/URL. Không dùng dự đoán của model làm nhãn kiểm tra.
5. Chia train/val/test theo người và buổi/nguồn từ trước. `group_id` phải giữ chung các ảnh
   liên quan, kể cả cùng người ở hai buổi. Không chia ngẫu nhiên các frame gần nhau qua ba tập.

```powershell
.venv\Scripts\python scripts/prepare_hazard_dataset.py --manifest data/detector/annotations.csv --out data/detector/hazards_v1 --task segment
.venv\Scripts\python scripts/package_hazard_dataset.py --source data/detector/hazards_v1 --out outputs/childsafe_hazards_v1.zip
```

Script kiểm tra nhãn hoàn chỉnh, tọa độ, ảnh trùng nội dung và group trùng split. Chỉ các lớp
thực sự có mẫu ở cả ba tập được đưa vào model; ID tự rút gọn theo tên trong `dataset.yaml`.
Việc kiểm tra tự động không xác minh người gán nhãn đúng hay đủ đa dạng; cần review độc lập.

## Train trên Kaggle rồi dùng trong app

Upload ZIP thành Dataset riêng, thêm vào `notebooks/childsafe_hazard_detector_kaggle.ipynb`.
Notebook có sẵn train 60 epochs, 960 px, batch 8, backbone local. Nếu thiếu VRAM giảm batch;
không thay ngưỡng theo test. Validation chọn checkpoint; test chỉ đánh giá sau train.
Notebook chưa được chạy trên Kaggle trong lượt sửa này.

Tải `childsafe_hazard_detector_results.zip`, giữ `evaluation.json`, `results.csv`, `weights/best.pt`
và nguồn/license dữ liệu. Đặt checkpoint tại `models/hazard_objects.pt`, sau đó:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_web_demo.ps1 -ObjectModel models/hazard_objects.pt
```

Không có trọng số mới thì chạy lệnh khởi động cũ; không tự tải hay chọn model bút cũ có kết quả yếu.
Model bbox cũng ghép được nhưng chỉ tạo CAUTION, không đổi thành segmentation khi nạp.

## Thử lại trên camera

- Lưu ba tệp phiên hiện tại trước khi tắt/khởi động lại server; phiên đang chạy chưa tự nhận mã mới.
- Bắt đầu ngang mặt, đủ sáng, gần; ghi kích thước mặt/vật bằng pixel ảnh gốc. Sau đó tăng khoảng
  cách và đổi góc từng bước, giữ cả các lần thất bại. Không có khoảng cách 2 m đảm bảo thành công.
- So sánh cùng đạo cụ/cảnh trước và sau. Nhãn độc lập theo hành động, không theo màu UI.
  Thu test mới sau khi chốt cấu hình; xem `docs/targeted_camera_trials.md`.
- Đo recall sự kiện tổng (bao gồm mất quan sát), cảnh báo nhầm/giờ, độ trễ, FPS và coverage.
  CAUTION/DANGER phải có báo cáo riêng. Chưa có các số này sau sửa thì chưa gọi là sẵn sàng triển khai.

Không ghi video mặc định, không đưa ảnh camera lên đám mây. Việc upload dữ liệu Kaggle do nhóm
thực hiện với dữ liệu đã có quyền sử dụng; không upload ảnh riêng tư từ thử nghiệm một cách tự động.
