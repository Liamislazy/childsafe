# ChildSafe: thử camera góc cao

> Runtime sản phẩm đã cập nhật ngày 12/10/2026: xem [hướng dẫn mới](product_runtime_guide.md).
> Crop có thể phóng to để tìm mốc, nhưng quality gate dùng pixel gốc. Classifier/tay
> không thay mask định vị; ảnh hết hạn trong lúc inference không thêm nguy cơ mới.

Ngày triển khai: 07/10/2026. Đây là runtime nghiên cứu, chưa được hiệu chuẩn hoặc nghiệm thu bằng webcam góc cao. Không thay thế giám sát của người lớn.

## Hành vi và giới hạn

Một nguồn webcam/URL, tối đa bốn người được phát hiện đồng thời. ID tạm thời dựa trên overlap; giao nhau, ghép gần bằng nhau hoặc vượt công suất làm mất lịch sử liên quan. ID không phải nhận dạng danh tính. Có thể có thêm dòng kết thúc cho ID vừa bị loại trong cùng timestamp.

Nguy cơ `normal/caution/danger` tách khỏi `observable/degraded/unobservable`. Xanh chỉ xuất hiện khi normal và observable. Cấu hình mặc định `quality.calibrated=false`, detector pretrained có `complete_object_coverage=false`; do đó giao diện luôn nêu hạn chế. Không chuyển hai cờ này chỉ để giao diện xanh. Chỉ xác nhận calibration sau thử trên dev góc cao; coverage chỉ khi các lớp đích đã có detector và được kiểm chứng.

MediaPipe nhận crop người không resize; landmark x/y được cộng offset về ảnh gốc, z giữ thang pixel crop và không dùng suy ra 3D. Mask YOLO phải cùng kích thước ảnh gốc. Kiểm tra ma trận hướng mặt, kích thước mặt/tay/miệng, biến động mốc, độ nhòe và phơi sáng. Đây là proxy chất lượng chưa hiệu chuẩn, không chứng minh miệng không bị che; không phát hiện được mọi kiểu che khuất.

Khoảng cách mask vật tới vùng miệng là khoảng cách tương đối trên ảnh chia chiều rộng mặt. Chỉ mask đủ tin cậy, miệng quan sát được và liên kết tay/vật rõ ràng mới tích lũy bằng chứng danger. Box, tay gần đầu và tọa độ giữ lại chỉ tạo nghi ngờ. Không đo centimet hoặc kết luận vật đã vào miệng. Khi nhiều vật được gán cho cùng người, xét vật có mask gần miệng nhất; đổi vật không overlap sẽ reset lịch sử. Chưa có re-identification vật khi bị che lâu.

Giữ nguy cơ gần nhất tối đa 0,8 giây khi mất miệng; không tăng dwell từ bằng chứng giữ lại. Mất quan sát dài, rời khung và nguồn ngắt kết thúc sự kiện với lý do khác nhau. Cooldown chỉ chặn notification, không hạ danger. Mất model hiển thị hạn chế và xóa classification cache.

Temporal tích phân các khoảng true-to-true trong cửa sổ 10/30 giây, cần 3/30 giây, không nối khoảng mất frame >0,25 giây. Dwell tích lũy từ bằng chứng tin cậy mới sau xác nhận cửa sổ. Tọa độ cũ không tính thời gian. N-of-M chỉ còn là primitive legacy; runtime không đếm frame. Tốc độ quá chậm được log; không đổi ngưỡng theo FPS.

## Chạy cục bộ

Python 3.12, Windows: cài theo `requirements-runtime.txt`, sau đó `pip install -e .`. Không dùng các phiên bản mới tự động cho benchmark so sánh. GPU chưa được kiểm thử trong lượt này; giữ dependency GPU riêng nếu cần.

Đặt trọng số local tại `models/yolo11n-seg.pt`, các bundle MediaPipe tại các đường dẫn trong config; classifier qua `--ckpt`. Xem `model_provenance.json` để kiểm tra SHA-256 bằng `Get-FileHash -Algorithm SHA256`. Model thiếu/hỏng không tự tải và không hiện xanh. Detector tùy chỉnh dùng YAML `_base_: default.yaml` với `detector.weights`; kiểm tra source URL/checksum riêng và giữ calibration=false cho đến khi kiểm chứng.

```powershell
python scripts/run_webcam.py --camera 0 --ckpt outputs/runs/efficientnet_b0_group/best.pt --log-features --condition elevated_dev
```

Với URL camera, đặt URL trong biến môi trường riêng rồi dùng `--source-env CHILDSAFE_VIDEO_URL`. Không dán URL chứa mật khẩu vào lệnh, config, tên condition, Git hoặc báo cáo. Backend xử lý trên máy; không upload frame/video. App không ghi video, chỉ event CSV và feature CSV khi yêu cầu. Feature CSV chứa tọa độ/ID tạm thời; vẫn cần quản lý quyền truy cập và thời hạn giữ. Các collector/Kaggle scripts có thao tác mạng nhưng không được gọi trong runtime.

Nguồn chạy thread với slot tối đa một frame. Timestamp là thời điểm nhận nội bộ; `frame_age_s` trước xử lý và `inference_s` không phải độ trễ đầu cuối. Frame trễ/lỗi không chạy model hoặc tạo bằng chứng mới. Reconnect/khôi phục sau stall reset lịch sử và đổi runtime session; session ID của capture được dùng để phát hiện reconnect ngay cả khi slot ghi đè thông báo ngắt. Nếu webcam backend chặn read, thread không thể bảo đảm shutdown tức thì.

## CSV và replay

Schema v2: `schema_version,person_id,session_id,t,received_t,frame_age_s,inference_s`, các landmark/khoảng cách cũ, `object_d_px,object_d_relative,mask_reliable,proximity_evidence`, hướng mặt, sharpness/brightness, mouth_visible, association_ambiguous, model/config/coverage flags, quality_reasons, obj_class/obj_conf/p_unsafe_raw/p_unsafe_cal/p_unknown_cal. Khoảng không có người vẫn có dòng person_id=0; không loại khỏi mẫu số thời gian. `history_reset` dùng cho đổi vật hoặc liên kết mơ hồ. Event CSV có `person_id,end_reason`.

`replay` dùng cùng AlertProcessor, tách từng người, đổi session xóa toàn bộ lịch sử. `object_present` phân biệt mất vật và classification chưa chạy; mất vật không làm mới classification cache. CSV cũ là một người và trả “không có dữ liệu chất lượng quan sát”; coverage=NaN. Replay CSV v2 cần đủ các cờ chất lượng; giá trị boolean thiếu không được coi là bằng chứng rõ.

Annotations nhiều người nên có `person_id`; nguồn reconnect phải có `session_id`, cùng clock `t` với features. Nếu thiếu person_id, báo cáo chỉ đánh giá sự kiện cấp camera, không xác nhận đúng người. Người có annotation nhưng không được phát hiện vẫn tính missed event trong recall tổng. Chỉ tune `--split dev`; `test/all` bị từ chối cho grid.

Report có recall tổng, FA/giờ, latency median/p90, FPS, coverage theo thời gian và recall trên sự kiện có toàn bộ mẫu trong đoạn được observable, không có gap >0,25 giây. Conditional recall chỉ bổ sung; không thay recall tổng. Coverage đòi hai đầu khoảng đều observable; gap dài tính không quan sát. FA/h khi annotations theo người dùng person-hours; cấp camera dùng camera-hours. FPS tính timestamp nguồn được xử lý, không nhân số dòng người. Thời gian inference/frame age là nội bộ. Model pretrained/cấu hình chưa hiệu chuẩn có coverage observable thấp hoặc 0 là kết quả phải giữ.

```powershell
python scripts/extract_features.py --videos data/recordings --ckpt outputs/runs/efficientnet_b0_group/best.pt
python scripts/evaluate_alerts.py report --split test
python scripts/benchmark_runtime.py --detector models/yolo11n-seg.pt --ckpt outputs/runs/efficientnet_b0_group/best.pt --face-model models/face_landmarker.task --hand-model models/hand_landmarker.task --samples 10
```

## Ma trận nghiệm thu webcam

Chỉ dùng người lớn đồng ý tham gia hoặc ma-nơ-canh; đạo cụ tù, quá lớn để nuốt; không cho vật vào miệng. Không quay trẻ em. Chạy dev trước, giữ các người/vật/phiên test riêng.

| Trục | Điều kiện phải thử | Trạng thái hiện tại |
|---|---|---|
| Camera | ngang, góc cao | chưa thử webcam |
| Mặt | thẳng, nghiêng, quay lưng/chỉ đỉnh đầu | logic tổng hợp đã thử; thực tế chưa thử |
| Độ phân giải | 480p, 720p, 1080p | chưa thử webcam |
| Ánh sáng | đủ sáng, thiếu sáng | logic blur/exposure đã thử; thực tế chưa thử |
| Số người | một, hai, bốn; vượt công suất | tracker/pipeline tổng hợp đã thử; thực tế chưa thử |
| Liên kết | người giao nhau, vật giữa hai người, đổi ID | logic tổng hợp đã thử; thực tế chưa thử |
| Che khuất | ngắn ≤0,8s, dài; chưa từng thấy miệng | logic tổng hợp đã thử; thực tế chưa thử |
| Vật | vật dài gần miệng khi tay xa; thiếu mask; pen/coin/vật lạ | logic tổng hợp đã thử; pretrained không bao phủ pen/coin |
| Nguồn | frame cũ, read lỗi, stall/reconnect, cooldown | fake source/logic đã thử; camera mạng chưa thử |

Mỗi tổ hợp cần ghi nhận dev/test, session/person ID, nhãn sự kiện, số missed/false alert, tổng thời gian kể cả đoạn mất quan sát, FPS, latency và coverage. Tổ hợp chưa chạy phải ghi “chưa thử”. Benchmark đơn ảnh không thay cho ma trận này. Xem `runtime_validation.md` cho kết quả thực tế của lượt triển khai.
