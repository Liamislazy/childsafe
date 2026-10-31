# Kiểm tra chẩn đoán bằng video công khai — 21/10/2026

## Nguồn và phạm vi

Đã tải ba clip người lớn từ Pexels qua trình duyệt, tổng 37.971.305 byte
(khoảng 38 MB thập phân). Giấy phép: [Pexels License](https://www.pexels.com/license/).
Chỉ dùng để suy luận local và tìm lỗi; chưa đưa vào train, không suy ra người
trong video sử dụng hoặc chứng thực ChildSafe. Không đưa clip/ảnh vào Git.

| ID | Clip gốc | Tác giả | Thời lượng file |
|---|---|---|---:|
| 7710699 | [A Woman Biting a Pen](https://www.pexels.com/video/a-woman-biting-a-pen-7710699/) | Kaboompics.com | 8,68 s |
| 6654092 | [Upset Man Rubbing Face](https://www.pexels.com/video/upset-man-rubbing-face-6654092/) | Alena Darmel | 8,56 s |
| 8901141 | [A Woman Scratching her Head with a Pen](https://www.pexels.com/video/a-woman-scratching-her-head-with-a-pen-8901141/) | Kaboompics.com | 18,24 s |

SHA-256 theo thứ tự trên:

```text
10be82b22d59882d0029141da4bab23037aa5d4b0eaade69b8cbe0b508fa2fda
195c7a2eefa6fb7f3918d700970f68d6c58711820d9ab1a7972bd7331413e38e
d8512caa4300c70284d3ef0d239f5116eb57615ee5b9d0fa50ef59cc7c34d068
```

## Cách chạy

Mã nguồn runtime ở commit `2304968`; không sửa rule hoặc model để chạy thử.
YOLO11n-seg local và classifier efficientnet_b0_group_robust 11 lớp, hash giữ
trong JSON từng lần chạy. Config mặc định, riêng CPU đặt 2 luồng. Thu nhỏ có giữ
tỷ lệ nếu chiều ngang nguồn lớn hơn 1280 px; không cắt riêng mặt hoặc tay trước
khi đưa vào pipeline. File nguồn 25 FPS, lấy mẫu khoảng 10 FPS theo timestamp
nguồn (khoảng lấy mẫu thực 0,08/0,12 s). Giữ deadline xử lý 250 ms và toàn bộ
ngưỡng quan sát, không dùng ảnh quá hạn làm bằng chứng mới.

Mỗi clip reset lịch sử người/cảnh báo. Đây là xử lý offline, không có hàng đợi
camera/browser, mạng truyền camera hoặc thời điểm chụp sensor thật. Thời gian đo
gồm khởi tạo lạnh MediaPipe và các biến động tài nguyên máy, không phải cam kết
FPS hay phép so sánh hiệu năng được kiểm soát. Chưa xác định khoảng cách người
tới camera trong các clip; không gọi đây là bài test 2 m hoặc camera góc cao.

File số liệu, mã chạy thử, cấu hình và nguồn được giữ local tại
`outputs/public_video_probe/`, không nằm trong Git. Các thư mục kết quả có
`features.csv`, `live_states.csv`, `replay.csv`, `report.json`, `review.jpg`.
`sources.json` lưu URL, tác giả, giấy phép và checksum; `videos/` chứa bản sao
các clip để dùng lại. Ảnh review là các mốc 1 giây, chỉ để xem động tác, không
phải nhãn hộp/mask hoặc nhãn sự kiện chính xác theo thời gian.

## Kết quả đã quan sát

| Clip | Mẫu xử lý | Mẫu hợp lệ theo cờ frame của mọi dòng người | Mẫu CAUTION/DANGER | Median/p90 xử lý |
|---|---:|---:|---:|---:|
| Sờ mặt | 86 | 23 | 0 / 0 | 261,8 / 291,6 ms |
| Cắn bút | 87 | 49 | 0 / 0 | 246,6 / 275,7 ms |
| Bút gãi đầu | 183 | 58 | 0 / 0 | 256,0 / 275,1 ms |

Tổng 356 mẫu, 130 mẫu hợp lệ theo tiêu chí trên. Không tính mỗi dòng người là
một frame. Replay CSV cho cùng mức nguy cơ tổng hợp với chạy offline ở tất cả
356 mẫu; điều đó chỉ xác nhận tính nhất quán phần mềm, không xác nhận dự đoán đúng.

- **Clip cắn bút:** các ảnh review cho thấy bút sát vùng môi, nhưng toàn clip không
  tạo cảnh báo. Có 0 dòng đạt điều kiện thấy miệng và chỉ 2 dòng có tay. Model
  bỏ sót mốc trong nhiều ảnh; các ảnh có mốc mặt cũng có lý do hướng mặt không
  phù hợp. Một dòng gán vùng vật thành baseball bat. Không thể giải thích lỗi
  chỉ bằng ngưỡng khoảng cách vật–miệng: thiếu cả định vị vật và mốc quan sát.
- **Clip sờ mặt:** không tạo cảnh báo, nhưng phần lớn mẫu bị loại/quá hạn nên
  không được lấy kết quả này làm bằng chứng đã giải quyết báo nhầm. Một dòng
  gán vùng vật thành tie; không phải nhãn vật đã được người kiểm tra xác nhận.
- **Clip bút gãi đầu:** không tạo cảnh báo; có 77 dòng định vị vật với nhãn
  toothbrush, cho thấy dự đoán loại vật vẫn sai dù không phát sinh cảnh báo.
- Lý do `frame expired during inference` xuất hiện nhiều. Các median của hai
  clip âm tính vượt 250 ms trong lần chạy này. Thiếu thời gian xử lý làm mất
  bằng chứng, không được diễn giải thành không có nguy cơ. Một số CSV có lý do
  lặp lại khi giữ lịch sử mất dấu; không đếm số chuỗi lý do như số frame lỗi.

Chưa gán nhãn temporal độc lập đầy đủ, chưa có hộp/mask ground truth, chỉ ba clip
được chọn để tìm lỗi nên không công bố recall, báo nhầm/giờ hay độ chính xác
tổng quát. Không gọi clip này là dữ liệu nghẹn.

## Hướng xử lý từ kết quả này

1. Dùng các video công khai phù hợp làm tập chẩn đoán có nguồn, thay vì chỉ dựa
   vào unit test hoặc ảnh đơn. Giữ nguyên các lỗi đã thấy trong báo cáo.
2. Đo và giảm chi phí detector/landmarks trên máy trình diễn; thử cấu hình CPU
   trên dev, không tăng giới hạn tuổi frame để che lỗi hiệu năng.
3. Cần dữ liệu bút/bút chì có hộp hoặc mask và ảnh âm tính sờ mặt/gãi đầu để train
   detector riêng. Tách người/nguồn video giữa train và test; các clip đã dùng
   để xem lỗi/chọn hướng sửa là dev, không gọi lại là test độc lập.
4. Kiểm tra trường hợp mặt nghiêng/che khuất riêng. Khi không thấy miệng, giữ
   trạng thái thiếu quan sát; không suy ra tiếp xúc mới từ mốc miệng cũ.
5. Đánh giá camera thật với khoảng cách đã đo, ánh sáng/độ phân giải có ghi nhận
   và nhãn sự kiện độc lập trước khi kết luận đạt yêu cầu ở 2 m.
