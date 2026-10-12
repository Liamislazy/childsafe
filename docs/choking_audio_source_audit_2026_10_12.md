# Kiểm tra dữ liệu âm thanh cho nhận diện nghẹn: 12/10/2026

**Chưa có dữ liệu đủ để train và xác nhận trẻ nghẹn do đồ chơi theo mục tiêu của
nhóm.** Nhãn bất thường âm thanh phải giữ riêng ý nghĩa, đối tượng và cảm biến.

## COUGHVID: kiểm tra metadata thật

Nguồn [Zenodo v3](https://zenodo.org/records/7024894). API công khai ghi giấy phép
`cc-by-4.0`; ZIP 2.297.542.075 byte, MD5 chủ nguồn công bố
`7c6cdf184748a2600538c331ba0ba718`. Không tải toàn bộ ZIP/âm thanh, không kiểm
chứng MD5 toàn bộ archive. Chỉ đọc directory và entry
`coughvid_20211012/metadata_compiled.csv` bằng HTTP Range: tổng 8.856.939 byte.
ZipFile kiểm tra CRC entry; CSV 5.463.841 byte có SHA-256:

`55b5ab27e1b1c9d4b97efa8b2bc39810492cf05cd9a6cfc157655706f0d343de`.

Kết quả `scripts/audit_audio_labels.py` trên CSV này:

| Chỉ số metadata | Bản ghi |
|---|---:|
| Tổng bản ghi | 34.434 |
| Có checkbox choking từ ít nhất một chuyên gia | 2.890 |
| Ít nhất một chuyên gia đánh dấu true | 27 |
| True kèm quality good/ok của cùng reviewer | 26 |
| Chỉ một reviewer, đánh dấu true | 23 |
| Reviewer bất đồng true/false | 4 |
| Ít nhất hai reviewer và tất cả cùng true | 0 |
| Tất cả checkbox hiện có false | 2.863 |
| Không có nhãn checkbox | 31.544 |

| Tuổi tự khai | Tổng bản ghi | Ít nhất một true | True với quality good/ok |
|---|---:|---:|---:|
| Dưới 18 | 1.285 | 1 | 1 |
| Từ 18 | 18.111 | 18 | 17 |
| Không có tuổi | 15.038 | 8 | 8 |

Reviewer 1 có 2 checkbox true, reviewer 2 có 25, reviewer 3/4 có 0. Không suy ra
reviewer nào đúng/sai từ số lượng này; cần kiểm tra phân bố reviewer và nhãn.

[Bài COUGHVID](https://arxiv.org/html/2009.11644v1) mô tả checkbox là nhận xét
âm thanh, đánh dấu khi nghi ngờ hợp lý; UUID nhận diện bản ghi. **Suy luận:**
nhãn không xác nhận dị vật đồ chơi hoặc tắc đường thở ở trẻ. False checkbox
không phải chẩn đoán âm tính; thiếu nhãn không phải âm tính. Tuổi tự khai không
xác nhận cohort; UUID không đủ để split theo người. Chưa thiết lập onset/offset
nghẹn hoặc video phòng đồng bộ.

Có thể khảo sát nhánh âm thanh sau khi làm rõ nhãn/danh tính. Không dùng các
checkbox để công bố recall nghẹn do đồ chơi hoặc ánh xạ sang `diagnosis=choking`.
Không mở âm thanh trong đợt này; không train nghẹn trẻ từ 27 checkbox.

## Nguồn khác

- [Smarty4Covid: bài tác giả](https://www.biosim.ntua.gr/file/get/papers/phpEoBWBz.pdf)
  có `audible_choking`, thu từ người trên 18 tuổi bằng điện thoại.
  [Zenodo](https://zenodo.org/records/8301142) hiện hạn chế truy cập, yêu cầu xét
  mục đích và thỏa thuận ký với chủ nguồn. Chưa có quyền/chưa đọc nhãn thật;
  không gửi yêu cầu thay mặt nhóm.
- [AudioSet cough](https://research.google.com/audioset/ontology/cough_1.html)
  có lớp tiếng ho, không tự cung cấp nhãn tắc đường thở do đồ chơi.
- [Aria](https://github.com/rishith-c/aria-listens) tự mô tả là prototype hackathon
  train âm thanh tổng hợp, ngưỡng chỉnh cho demo. Không lấy demo/model này làm
  chứng cứ y khoa cho hệ thống của nhóm.

## Điều kiện đi tiếp tới model đúng mục tiêu

1. Chuyên gia định nghĩa nhãn tắc đường thở, âm thanh bất thường, gagging và ho;
   tách xác nhận ca khỏi nhận xét chỉ dựa vào âm thanh. Xác minh nguyên nhân đồ
   chơi riêng, không suy ra từ vật từng xuất hiện gần miệng.
2. Dữ liệu hợp lệ phù hợp camera phòng + micro, gồm ca có/không có âm thanh và
   nhãn diễn biến độc lập. Việc đánh giá ở trẻ cần hợp tác/quy trình riêng;
   workflow repo vẫn không xử lý video trẻ thật. Không tạo nghẹn để lấy dữ liệu.
3. Tách người/phiên/site giữa train-dev-test, tránh trùng người/bản ghi; UUID
   bản ghi không chứng minh các người độc lập.
4. Train model thời gian AV, hiệu chuẩn dev, đánh giá test độc lập; tách kết quả
   simulation khỏi ca được xác nhận. Diễn lại không chứng minh hiệu quả ở trẻ.
5. Đo từng góc/vị trí/độ phân giải/ánh sáng/âm nền, mất cảm biến và người giao
   nhau. Điều kiện chưa thử ghi chưa đánh giá. Camera không thấy đường thở;
   bao phủ hình ảnh không tự chứng minh đã đo được thông khí.

## Kiểm tra hoàn thành mục tiêu

| Yêu cầu thật | Bằng chứng hiện có | Kết luận |
|---|---|---|
| Trẻ đang nghẹn | Chưa có model phù hợp/test lâm sàng | Chưa đạt |
| Nghẹn do đồ chơi | Không có nhãn nguyên nhân độc lập | Chưa đạt |
| Hình ảnh + âm thanh trên thiết bị | Có timeline; chưa có producer/model AV | Chưa đạt |
| Mọi nơi/mọi góc, edge cases | Có ma trận; chưa đánh giá đại diện | Chưa đạt |
| Đồng bộ/thiếu dữ liệu/reconnect | 14 test số tổng hợp | Chỉ kiểm chứng phần mềm |
| Recall/cảnh báo nhầm/độ trễ nghẹn | Có evaluator; chưa có prediction thật | Chưa đạt |

Thiếu chuyên gia/ground truth phù hợp đã tồn tại qua các lượt xây evaluator,
timeline và kiểm tra nguồn. Train bằng nhãn thay thế không chứng minh nghẹn trẻ
thật. Bước tiếp theo cần đầu vào chuyên môn/quyền dữ liệu từ bên ngoài; không
bật chẩn đoán từ mô hình chưa xác nhận.
