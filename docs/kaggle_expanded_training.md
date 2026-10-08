# Huấn luyện bổ sung trên Kaggle

## Vấn đề đã xác định

Checkpoint đang chạy có 10 lớp và không có `pen` hoặc `coin`. Một vật ngoài các lớp đã học vẫn bị softmax ép vào một lớp quen thuộc. Không thể suy ra chính xác nguyên nhân mất cảnh báo chỉ từ một ảnh tĩnh: còn có thể mất mốc tay/miệng hoặc vật bị che.

Trong giao diện cũ, `p` là xác suất **nhóm unsafe**, không phải xác suất của lớp vật. Code cũ dùng độ tin cậy của nhóm safe/unsafe mà bỏ qua `obj_conf`. Code mới yêu cầu cả hai vượt ngưỡng, coi `other`/`unknown` là chưa xác định, và giữ trạng thái gần trong một khoảng ngắn khi mất mốc. Đây không phải cơ chế phát hiện mọi vật lạ: mô hình vẫn có thể nhận sai với độ tin cậy cao.

## Những file đã chuẩn bị

- `notebooks/childsafe_expanded_kaggle.ipynb`: notebook tự chứa code, cấu hình và bảng chia ChildSUn gốc; không cần upload cả repository.
- `outputs/datasets/childsafe_supplement.zip`: ảnh bổ sung, manifest chứa nguồn/giấy phép/tác giả, bảng ảnh để kiểm tra và báo cáo thu thập.
- `scripts/collect_openimages.py`: thu thập ảnh vật có bounding box từ Open Images. Không cào ảnh từ kết quả tìm kiếm.
- `scripts/prepare_expanded_data.py`: hợp nhất dữ liệu đã duyệt, kiểm tra ảnh trùng và chia theo nhóm nguồn.
- `scripts/smoke_expanded_training.py`: kiểm tra luồng train/lưu/nạp lại trên CPU; checkpoint này **không được dùng triển khai**.

Code huấn luyện được sửa từ pipeline hiện có của nhóm: giữ EfficientNet-B0/MobileNetV3 và định dạng checkpoint để dùng lại trong webcam.

## Dữ liệu hiện có

Ngày 07/10/2026 tải 194 ảnh: bút 34, đồng xu 24, dao 57, kéo 1, nĩa 33, thìa 45. Sau xem trực quan các bảng ảnh, duyệt 169: bút 29, đồng xu 23, dao 56, kéo 1, nĩa 25, thìa 35. Loại 25 ảnh mơ hồ, quá mờ hoặc nguồn chứa nhiều loại vật trong bộ lấy mẫu này. Các ảnh loại vẫn được giữ để kiểm tra nhưng không vào train.

Đây là **bộ khởi đầu nhỏ**, không phải bộ dữ liệu chứng minh sản phẩm hoạt động. Số ảnh kéo thấp do điều kiện lấy mẫu; lớp kéo vẫn có dữ liệu ChildSUn. Các ảnh bút cận ngòi, thân bút hoặc bút đánh dấu có thể khác nhiều so với webcam. Số ID ảnh khác nhau không bảo đảm là các vật lý khác nhau; kiểm tra ảnh gần trùng giúp giảm rò rỉ nhưng không thay thế kiểm tra theo vật và phiên ghi hình.

Nguồn: [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html), [kho ảnh CVDF](https://github.com/cvdfoundation/open-images-dataset). Manifest lưu giấy phép CC BY của từng ảnh, URL gốc, tác giả và bounding box. Ảnh đã được cắt theo bounding box, thêm lề 8% và lưu JPEG; khi chia sẻ phải giữ thông tin ghi công và thông báo đã chỉnh sửa. Bộ lọc loại nguồn có nhãn người/khuôn mặt, sau đó duyệt ảnh crop; thiếu nhãn người không tự nó chứng minh ảnh không có người.

`unsafe` của bút/đồng xu là **quy ước nguy cơ trong ngữ cảnh đưa vật vào miệng**. Nó không có nghĩa mọi hoạt động dùng bút đều nguy hiểm. `safe` của thìa/trái cây kế thừa bài toán gốc, không chứng minh ăn uống an toàn hay không nghẹn. Không tự gán vật chưa biết thành safe.

## Các bước trên Kaggle

1. Tạo Kaggle Dataset riêng tư từ ZIP bổ sung; thêm nó cùng dataset ChildSUn vào notebook.
2. Import `childsafe_expanded_kaggle.ipynb`. Bật GPU và Internet để tải thư viện/trọng số ImageNet.
3. Sửa `RAW_ROOT` thành thư mục chứa trực tiếp các tên ảnh trong bảng chia gốc; sửa `SUPPLEMENT` thành đường dẫn `manifest.csv` trong input bổ sung. Không chọn tự động thư mục lớn nhất.
4. Bộ khởi đầu đã có `reviewed=true` cho 169 ảnh. Kiểm tra lại bảng ảnh, sửa nhãn hoặc đặt false nếu thấy không phù hợp. Ảnh mới tải mặc định luôn false.
5. Chạy phần chuẩn bị. Test ChildSUn giữ nguyên. Validation gốc được chia theo nhóm thành validation chọn checkpoint và calibration fit temperature. Ảnh bổ sung được chia theo nhóm nguồn, khoảng 70/10/10/10 cho train/val/cal/test.
6. Nếu phát hiện ảnh trùng nối hai tập gốc, nhãn mâu thuẫn hoặc lớp mới không đủ nhóm, chương trình dừng để kiểm tra; không được bỏ qua lỗi bằng random split. Bảng `expanded.report.json` ghi số ảnh và ảnh gần trùng. Mọi lớp phải có mẫu ở cả bốn tập trước khi train.
7. Train mặc định EfficientNet-B0, 30 epochs tối đa, early stopping, trọng số loss theo căn nghịch đảo tần suất lớp, augmentation che một phần/nhòe/ánh sáng và letterbox giữ cả vật dài. Không dùng random crop mạnh làm biến mất đầu bút. Checkpoint ghi preprocessing để inference dùng đúng phép biến đổi.
8. Tải ZIP kết quả. Lưu `run_config.json`, `training_manifest.csv`, `metrics.json`, `logits.npz`, `best.pt`. Chạy webcam với `best.pt` mới. Phiên webcam đang mở không tự nạp code/checkpoint mới: đóng bằng Q rồi mở lại.

Các phân hoạch `validation`/`test` trong tên nguồn Open Images chỉ là tên nguồn tải. Chúng không được dùng làm tập kiểm tra dự án một cách mặc định; bảng chia nhóm riêng quyết định vai trò từng ảnh.

## Bổ sung dữ liệu sát thực tế hơn

Mục tiêu thu thập tiếp theo, **chưa phải dữ liệu đã có**: mỗi lớp bút/đồng xu/nắp nhỏ có 100–200 ảnh từ nhiều vật thật và 5–10 phiên/người lớn đồng ý tham gia. Chụp crop đúng vùng tay mà ứng dụng sử dụng, ở xa/gần miệng, nghiêng, thiếu sáng và che khuất 25–75%. Dùng đạo cụ tù hoặc ma-nơ-canh để mô phỏng gần miệng; không cho vật vào miệng, không thử bằng trẻ em. Thu thêm tay rỗng và vật chưa biết để đánh giá từ chối nhận diện riêng; không mặc định đưa chúng vào lớp safe.

Có thể thêm dòng vào manifest với các cột `path,category,label,group,domain,reviewed`. `path` tính từ vị trí manifest. Đặt `domain=adult_hand_crop`. Giữ mọi ảnh của cùng người/phiên hoặc cùng vật thử trong một nhóm; người/vật dành cho kiểm tra không dùng để train. Nếu một người xuất hiện trong nhiều phiên, dùng nhóm người thay vì từng frame. Nhóm có thể chứa nhiều lớp. Mẫu nhóm phải đủ để cả bốn tập có các lớp cần đánh giá.

Collector hỗ trợ `--subset test` để lấy thêm ảnh từ nguồn Open Images khác. Đây là tùy chọn mở rộng, chưa được chạy/duyệt trong bộ khởi đầu. Khi hợp nhất, đặt ảnh và đường dẫn manifest về một thư mục chung; giữ nguyên `group` có prefix nguồn và duyệt thủ công. Đừng chạy lại collector vào thư mục đã duyệt nếu muốn giữ các chỉnh sửa manifest.

## Cách chứng minh cải thiện

So sánh baseline và bản cải tiến trên cùng tập kiểm tra gốc và cùng kịch bản webcam giữ riêng. Chọn kiến trúc/ngưỡng bằng validation, fit temperature bằng calibration; không chọn lại dựa trên test. Có thể chạy baseline 10 lớp trên ảnh bút/đồng xu để đo tỷ lệ nhận nhầm safe, nhưng không báo accuracy 12 lớp cho baseline không có hai lớp đó.

Đọc recall riêng của bút/đồng xu, confusion với fruit/spoon, unsafe recall, ECE, coverage và số unsafe bị kết luận safe với độ tin cậy cao. Phân biệt mẫu bị từ chối sang caution với mẫu bị nhận nhầm safe. Các test bổ sung hiện chỉ vài ảnh/lớp; chưa đủ để báo con số ổn định hoặc khẳng định vượt mô hình cũ.

Đối với sản phẩm và VISEF, đánh giá riêng **recall theo sự kiện, cảnh báo nhầm/giờ, thời gian cảnh báo và FPS** trên kịch bản mô phỏng người lớn/ma-nơ-canh. Kết quả phân loại ảnh không thay thế kết quả cảnh báo sự kiện; chưa có bằng chứng giảm tai nạn thực tế.

## Đã kiểm tra / chưa kiểm tra

Đã chạy kiểm thử logic từ chối nhận diện, replay, che khuất, giữ hai đầu vật dài và chống rò rỉ nhóm. Đã train thử một epoch từ khởi tạo ngẫu nhiên trên hai lớp crop thật và nạp lại checkpoint. Vòng thử này chỉ kiểm tra code, không đo chất lượng mô hình.

Chẩn đoán thêm trên 29 crop bút đã duyệt: checkpoint cũ với logic cũ cho safe 3, uncertain 8, unsafe 18. Giữ nguyên trọng số và áp dụng bộ lọc mới: safe 1, uncertain 20, unsafe 8. Đây là phép thử ảnh vật riêng lẻ, không phải điểm test của mô hình đã train mới hay đánh giá cảnh báo webcam. Bộ lọc làm tăng số trường hợp caution, và vẫn còn một trường hợp nhận nhầm safe. Chi tiết ở `outputs/verification/pen_baseline_report.json` và CSV dự đoán.

Chưa chạy train đầy đủ trên Kaggle, chưa kiểm tra hợp nhất với ảnh ChildSUn thật tại máy này (hiện chỉ có bảng chia), chưa đánh giá lại webcam bằng checkpoint mới. Notebook chưa được thực thi trong môi trường Kaggle thực.
> Ghi chú runtime góc cao (07/10/2026): các kết quả thu thập/train thử bên dưới được kế thừa từ checkout gốc, không chạy lại trong lượt triển khai runtime này. Không tải thêm dữ liệu hoặc train mới. Classifier bổ sung không định vị đầu bút/đồng xu; runtime cần detector có mask phù hợp. YOLO11n-seg pretrained chưa có các lớp này, nên vẫn hiển thị hạn chế quan sát.

