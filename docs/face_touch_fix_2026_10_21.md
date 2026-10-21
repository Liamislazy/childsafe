# Sửa cảnh báo khi sờ má, gãi mặt — 21/10/2026

## Thay đổi hành vi

- Tay gần mặt hoặc classifier nhận một crop tay là `knife`/`other` không đủ để khởi tạo cảnh báo.
  Giữ thông tin đó làm chẩn đoán kỹ thuật; không suy ra tay trống chắc chắn hoặc an toàn.
- Khoảng cách cảnh báo lấy từ vật đã định vị, không lấy giá trị nhỏ hơn giữa tay và vật.
  Vật còn xa trong khi tay chạm má không tạo cảnh báo; vật dài sát miệng khi tay còn xa vẫn được xét.
- Ứng viên box/mask yếu cần 0,15 giây bằng chứng qua các khoảng frame hợp lệ trước CAUTION.
  Đây là giá trị khởi đầu chưa hiệu chuẩn. Mask tin cậy/dwell vẫn bắt buộc cho DANGER.
- Chỉ tiếp tục cảnh báo đã bắt đầu từ vật. Mất miệng/vật giữ tối đa `lost_hold_s=0.8`;
  tay tiếp tục che mặt không gia hạn vô thời hạn và không khởi tạo sự kiện mới từ một lần thấy mơ hồ.
- Sửa replay đọc cột CSV trộn số và boolean (`1`, `0`, `True`, `False`, `1.0`), để giữ kết quả
  cùng với runtime. UI/CSV ghi rõ khi không có khoảng cách vị trí vật dùng cho cảnh báo.

## Dữ liệu phiên người dùng

File JSON người dùng gửi chỉ có metadata; đã lấy thêm features và nhãn từ đúng phiên còn lưu
trong RAM của server local 8001. Đã kiểm tra code/config khớp metadata trước khi sửa.
2.200 dòng, khoảng phiên 25,47–362,13 giây; một nhãn `non_event` 187,55–271,92 giây
(84,36 giây), không có nhãn sự kiện dương tính. Bản sao số liệu được giữ trong
`outputs/face_touch_fix/`, không commit hoặc gửi lên dịch vụ ngoài.

So sánh dưới đây là **phát lại cùng features**, không chạy lại model trên video, không phải test mới.
Cả luật cũ ở `c066ec2` và luật mới cùng dùng parser CSV đã sửa:

| Đại lượng | Trước | Sau |
|---|---:|---:|
| Dòng CAUTION trên toàn phiên | 385 | 169 |
| Trong đó không định vị được vật ở dòng hiện tại | 185 | 43 |
| Đợt CAUTION trong đoạn `non_event` 84,36 giây (merge 2 giây) | 6 | 6 |
| Recall sự kiện | Chưa đo | Chưa đo |

Không gọi các dòng không có vật là toàn bộ báo nhầm: có thể đang giữ bằng chứng gần nhất.
Số sơ bộ trước khi sửa parser (87 dòng cảnh báo không định vị vật) đã được thay bằng số đối chiếu
185 ở trên. Không ngoại suy độ ổn định của tỷ lệ mỗi giờ từ một đoạn âm tính ngắn.

**Chưa loại hết báo nhầm.** Trong đoạn âm tính, detector vẫn có các dự đoán bàn chải, điện thoại,
kéo và đồng hồ ở vùng tay/mặt. Giảm số dòng cảnh báo không chứng minh giảm số đợt báo sai;
số đợt ở đoạn này vẫn là 6. Cần ảnh đúng cảnh lỗi để kiểm tra mask và bổ sung hard negatives.

## Kiểm chứng

- Python: **269 passed, 3 skipped**, 17 cảnh báo dependency deprecation, 25,13 giây.
- Có regression tay trống nhiều vị trí, crop tay nhận nhầm thành dao, mất mặt sau một glimpse,
  vật dài/tay xa, box cần thời gian, occlusion ngắn/dài và CSV mixed flags live/replay.
- Edge headless: nút tải ảnh chỉ tải một PNG sau thao tác bấm; không có tải tự động hoặc webcam thật.
  Kiểm tra scope, nguồn detector và nền xám khi thiếu quan sát tiếp tục qua.
- Chưa có phiên camera mới có cả dương tính/âm tính độc lập sau sửa; không tuyên bố recall được giữ nguyên.

## Thu ví dụ còn lỗi

Trong demo bấm **Tải ảnh kiểm tra** ở đúng tư thế gây lỗi; ảnh nguyên khung không có overlay.
Ảnh chỉ được tải khi người dùng bấm; không tự lưu/video/upload. Ảnh tĩnh có thể khác một chút với
frame vừa suy luận nên nên giữ nguyên tư thế khi bấm. Ghi loại hành động và thời điểm tương ứng
trong CSV; dự đoán model không phải nhãn thật.

Đưa ảnh bàn tay trống/gãi má/chống cằm vào nhóm nền âm tính sau review đầy đủ. Với bộ 15 lớp,
ảnh không có bất kỳ vật mục tiêu nào có `.txt` YOLO rỗng; nếu có vật mục tiêu thì phải gán đúng.
Giữ các ảnh cùng người/buổi trong một split, giữ test độc lập. Xem quy trình Kaggle ở
`docs/expanded_hazard_runtime.md`. Chưa có ảnh được nhóm cho phép dùng thì không tự huấn luyện từ dữ liệu phiên.
