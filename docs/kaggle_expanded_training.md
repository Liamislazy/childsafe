# Train ChildSafe trên bộ dữ liệu đầy đủ bằng Kaggle

Import `notebooks/childsafe_expanded_kaggle.ipynb`. File `childsafe_kaggle.ipynb` cũng dùng cùng workflow đầy đủ; chỉ cần chạy một notebook. Notebook tự chứa source, cấu hình và bảng chia cố định.

## Dữ liệu

Đã tải và kiểm tra ngày 08/10/2026:

| Nguồn | Số ảnh/crop | Vai trò |
|---|---:|---|
| ChildSUn | 5.353 | Giữ train/test gốc; chia validation thành val và cal |
| Open Images train | 20.557 | Pool bổ sung, chia theo nhóm nguồn cố định |
| Open Images validation | 1.973 | Pool bổ sung, chia theo nhóm nguồn cố định |
| Open Images test | 4.069 | Đánh giá ngoài miền, không đưa vào train/val/cal |

Pool chính có **27.883 ảnh, 11 lớp**: `babyitems`, `ball`, `fork`, `fruit`, `heater`, `knife`, `other`, `scissors`, `socket`, `spoon`, `toy`. Dữ liệu đầy đủ nghĩa là dùng đủ pool đã chọn với holdout riêng; test không cập nhật trọng số. Đây là số crop lấy theo giới hạn script, không phải toàn bộ Open Images.

`other` là lớp để runtime từ chối, không phải khẳng định an toàn. Pen hiện thuộc `other`; coin không có lớp riêng. Nhãn binary `safe` trong bảng cũ chỉ dùng tính P(unsafe). Workflow 169 crop pen/coin đã duyệt còn ở `scripts/prepare_expanded_data.py`, nhưng không tự trộn vào bộ 11 lớp vì chính sách nhãn Pen khác nhau.

## 1. Upload dữ liệu

Tại thư mục repository, tạo ZIP chỉ gồm crop và manifest ghi công, bỏ metadata gần 3 GB:

```powershell
Compress-Archive -Path data\external\openimages\train,data\external\openimages\validation,data\external\openimages\test,data\external\openimages\manifest.csv -DestinationPath outputs\openimages_crops.zip
```

Nếu ZIP đã tồn tại, chọn tên mới để giữ bản cũ. Tạo Kaggle Dataset từ ZIP rồi Add Input vào notebook. Giữ manifest chứa nguồn/tác giả/giấy phép khi chia sẻ crop. Không upload `.venv`, weights hoặc metadata. Tổng crop khoảng 377 MB trước nén.

Thêm dataset ChildSUn `fairuzkhan002/datasetvgg19`. Input Open Images cần chứa:

```text
<OPENIMAGES_ROOT>/
  manifest.csv
  train/<category>/*.jpg
  validation/<category>/*.jpg
  test/<category>/*.jpg
```

Kaggle có thể giải nén ZIP trực tiếp ở root hoặc trong thư mục con; xem Input panel để chọn đúng đường dẫn.

## 2. Import notebook và sửa đường dẫn

Import notebook, bật GPU và Internet trong Settings. Internet dùng cài dependency và lấy pretrained ImageNet weights. Cell setup dừng nếu không có CUDA GPU.

Trong cell Input paths sửa:

```python
RAW_ROOT = Path('/kaggle/input/<childsun>/<thu-muc-chua-anh>')
OPENIMAGES_ROOT = Path('/kaggle/input/<openimages-dataset>/<root-neu-co>')
```

`RAW_ROOT` chứa trực tiếp filename dạng `baby_items_10_0_1720.jpg`. `OPENIMAGES_ROOT` chứa trực tiếp manifest và ba thư mục train/validation/test.

Chạy setup, các cell source rồi cell chuẩn bị. Script `prepare_full_training.py` đối chiếu mọi ảnh với `splits_with_other_train.csv`, kiểm tra nhóm nguồn và download manifest, rồi tạo `data/splits/full_training.csv` với đường dẫn tuyệt đối trên Kaggle. Nếu thiếu/thừa file, sửa input thay vì bỏ assertion hoặc âm thầm train tập con.

## 3. Split và calibration

Bảng gốc có train 19.523, validation 4.194, test 4.166. Giữ nguyên mọi ảnh train/test và các nhóm chống trùng đã có. Chia một nửa số nhóm validation sang `cal` bằng seed 42, yêu cầu mọi lớp có mặt trong cả val và cal. Số ảnh hai phần có thể lệch vì kích thước nhóm khác nhau. Crop cùng ảnh nguồn không đi qua nhiều phần.

`val` chọn checkpoint theo macro-F1 và early stopping; `cal` fit temperature; `test` đánh giá nội bộ. `test_by_domain` tách kết quả theo nguồn; ChildSUn test vẫn gồm 789 ảnh. Open Images test 4.069 crop được đánh giá riêng sau khi chọn cấu hình. Không chọn model/ngưỡng bằng cách thử lại trên test.

Notebook dùng bảng split đã commit, không hash lại toàn bộ dữ liệu trong phiên Kaggle. Khi đổi dữ liệu cần tạo và kiểm tra split mới, không dùng bảng cố định cũ.

## 4. Train

Mặc định EfficientNet-B0 pretrained, tối đa 30 epochs, batch 32, robust augmentation, letterbox 224×224 và sqrt WeightedRandomSampler. Letterbox giữ cả vật dài; checkpoint ghi preprocessing để inference dùng cùng phép biến đổi.

Dùng sampling sqrt, không bật thêm `--balanced` (loss weighting). Một epoch lấy số mẫu bằng kích thước train với replacement, nên không đảm bảo từng ảnh đều được lấy trong mỗi epoch; pool và loader vẫn bao gồm đủ ảnh train.

Có thể đổi `ARCH='mobilenetv3_large_100'`, giảm `BATCH_SIZE=16` khi thiếu VRAM hoặc đổi `EPOCHS`. Chọn `RUN` mới cho mỗi cấu hình để giữ kết quả cũ. Cell train sẽ dừng nếu `best.pt` đã tồn tại; pipeline hiện không hỗ trợ resume giữa epoch.

Lệnh notebook thực thi tương đương:

```bash
python -m childsafe.classifier.train --arch efficientnet_b0 --split group \
  --config configs/kaggle_full.yaml --manifest data/splits/full_training.csv \
  --robust --preprocessing letterbox --epochs 30 --batch-size 32 \
  --out outputs/full_dataset/efficientnet_b0_robust_letterbox
```

Không cần tải Open Images lại trên Kaggle. `--workers 8` của downloader là số luồng mạng; `num_workers=2` trong notebook là số worker DataLoader, có vai trò khác.

## 5. Kết quả và checkpoint

Đọc `metrics.json`: macro-F1, recall từng lớp, unsafe recall, ECE, rejection coverage và confusion matrix. Đọc `test_by_domain` để tránh tổng accuracy che khuất lỗi theo nguồn. Cell External Open Images test chạy `evaluate --ood`, tạo `eval_report.json`.

Cell cuối tạo `/kaggle/working/full_dataset_results.zip`, gồm:

- `best.pt` có temperature và preprocessing.
- `metrics.json`, `logits.npz`, `run_config.json`, `training_manifest.csv`.
- `eval_report.json` và các biểu đồ pipeline tạo.
- `full_training.csv`, báo cáo chuẩn bị, `kaggle_full.yaml`, manifest ghi công Open Images.

Tải ZIP qua link cuối notebook hoặc Output của phiên Kaggle. Manifest lưu đường dẫn tuyệt đối của phiên train; trên máy khác chạy lại chuẩn bị với root mới.

Chạy runtime local bằng checkpoint mới:

```powershell
.venv\Scripts\python scripts\run_webcam.py --ckpt <duong-dan-best.pt>
```

Phân loại ảnh chưa xác nhận chất lượng cảnh báo webcam góc cao/nhiều người. YOLO11n-seg pretrained thiếu pen/coin; cần detector phù hợp để định vị chúng. Nghiệm thu runtime theo `docs/runtime_validation.md`.

## Kiểm tra và tái tạo

Đã đối chiếu đủ 22.530 crop Open Images train/validation với bảng đầy đủ, không thiếu ảnh. Suite có 76 kiểm thử đều qua, gồm source nhúng trong cả hai notebook, chuẩn bị manifest/provenance, giữ test và chống rò rỉ calibration. Kiểm tra cú pháp script cũng qua. ChildSUn thật chưa có ở worktree này nên cell chuẩn bị sẽ kiểm tra lại input trên Kaggle; chưa chạy full GPU training hay phiên Kaggle thật trong lượt sửa này.

Khi sửa source pipeline, tái tạo cả hai notebook:

```powershell
.venv\Scripts\python scripts\build_expanded_kaggle_notebook.py
.venv\Scripts\python scripts\build_kaggle_notebook.py
```
