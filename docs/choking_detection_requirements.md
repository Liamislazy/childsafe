# Nhận diện nghẹn do đồ chơi: yêu cầu và tình trạng bằng chứng

Ngày nghiên cứu: 12/10/2026. Mục tiêu vẫn là nhận diện trẻ bị nghẹn sau khi nuốt/hít phải đồ chơi,
bao gồm nhiều vị trí, hướng nhìn và edge cases. **Mục tiêu này chưa đạt.** Tài liệu này không đổi mục
tiêu thành phát hiện vật gần miệng; nó ghi những điều phải chứng minh để đạt mục tiêu ban đầu.

## 1. Cơ sở cần phân biệt

- `unsafe_mouthing` trong runtime hiện tại là một suy luận vị trí vật/tay gần miệng trong ảnh.
  Đó không phải nhãn tắc đường thở, không chứng minh trẻ đã nuốt vật và không chứng minh nguyên nhân
  nghẹn là đồ chơi. ChildSUn/Open Images không cung cấp ground truth lâm sàng này.
- Hướng dẫn [Resuscitation Council UK 2025](https://www.resus.org.uk/cy/node/36439) mô tả nghi ngờ
  dị vật đường thở dựa vào khả năng nói/khóc và bối cảnh; biểu hiện phải được đánh giá, không đơn thuần
  bằng hình ảnh vật. [NHS](https://www.nhs.uk/best-start-in-life/baby/weaning/safe-weaning/choking-and-gagging-on-food/)
  phân biệt gagging và choking, mô tả choking có thể im lặng và khác biệt biểu hiện màu da.
  [RCUK BLS FAQ](https://www.resus.org.uk/cy/node/671) phân biệt tắc một phần và tắc nặng.
- Suy luận kỹ thuật từ các nguồn trên: im lặng, ho hoặc ôm cổ không được dùng riêng lẻ để xác nhận
  nghẹn; không ho không loại trừ nghẹn. Không yêu cầu ôm cổ như điều kiện bắt buộc ở trẻ nhỏ.
- Ngay cả giả định camera bao phủ toàn bộ căn phòng từ nhiều hướng, vẫn không trực tiếp nhìn thấy
  đường thở. Chuyển động ngực cũng không tự chứng minh có luồng khí. Xác nhận cơ chế tắc và nguyên
  nhân do đồ chơi cần nhãn tham chiếu độc lập từ chuyên gia/cơ sở lâm sàng.

## 2. Những phần phải phát triển và kiểm chứng

1. Theo dõi người và đồng bộ timeline giữa các camera; các ID tracker cục bộ không được coi là cùng
   người giữa camera hoặc sau reconnect nếu chưa có liên kết được kiểm chứng. Không nhận diện tuổi
   bằng đoán ngoại hình để quyết định an toàn.
2. Bộ nhận diện diễn biến theo thời gian, tách khỏi classifier loại vật. Bằng chứng về đồ chơi gần
   miệng là bối cảnh hỗ trợ, không phải điều kiện bắt buộc: một ca nghẹn có thể bắt đầu ngoài khung hình.
3. Âm thanh chỉ thêm nếu người dùng chọn, xử lý trên thiết bị. Kiểm tra microphone mất/tắt, TV,
   tiếng nói của người khác, nhiều trẻ và không chắc nguồn âm. Không mặc định im lặng là ngừng thở.
4. Tách kết quả cảnh báo nghi ngờ khỏi xác nhận y khoa. Chưa có model/đánh giá phù hợp thì hiển thị
   `choking assessment unavailable`, không `not choking` và không một xác suất bịa ra.
5. Mất quan sát, góc nhìn không hỗ trợ hoặc model ngoài miền phải có kết quả thiếu thông tin. Không
   tự xóa nghi ngờ nghẹn chỉ vì đồ chơi không còn được nhìn thấy hoặc người đứng im.
6. Mô hình, ngưỡng, tốc độ phản ứng và cách trình bày cảnh báo phải được chuyên gia đánh giá. Chưa
   đặt ngưỡng recall/false alarms chấp nhận được bằng suy đoán. Không chờ dấu hiệu đổi màu da làm
   điều kiện bắt buộc kích hoạt cảnh báo, không coi model phân biệt màu da là bằng chứng đo oxy.

## 3. Ma trận edge cases bắt buộc

| Nhóm | Các trường hợp phải có ground truth và đánh giá riêng |
|---|---|
| Biểu hiện | Không có tiếng ho rõ; tắc một phần; không ôm cổ; bất động hoặc thay đổi đáp ứng |
| Tiền sử vật | Chưa từng thấy đồ chơi; vật bị tay che; đồ chơi ngoài các lớp đã học; nguyên nhân chưa rõ |
| Hành vi dễ nhầm | Ho thông thường, gagging, khóc, cười, nuốt bình thường, ngậm đồ chơi, gãi cổ, ngủ |
| Góc/vị trí | Trước mặt, nghiêng, từ cao, quay lưng, sau đồ nội thất, nằm/ngồi/đứng, thay đổi camera |
| Hình ảnh | Người/vật nhỏ, nhòe chuyển động, nén mạnh, thiếu sáng, ngược sáng, chế độ hồng ngoại |
| Đối tượng | Trẻ sơ sinh/trẻ nhỏ/trẻ lớn; không suy rộng diễn lại của người lớn sang trẻ |
| Nhiều người | Tay của người chăm sóc, bế trẻ, người giao nhau, sai ID, âm thanh không thuộc người đang theo dõi |
| Nguồn dữ liệu | Camera/microphone mất, frame cũ, lệch đồng hồ, reconnect, bỏ sót diễn biến ban đầu |

“Mọi góc/mọi nơi” là một yêu cầu bao phủ phải chứng minh. Một nhóm góc không có dữ liệu phải ghi
`not evaluated`, không lấy điểm trung bình toàn bộ để coi nhóm đó đã đạt. Che hoàn toàn hoặc tín hiệu
không chứa thông tin cần thiết là giới hạn cảm biến, không khắc phục được chỉ bằng thêm lớp đồ vật.

## 4. Dữ liệu và đo lường

- Chưa tìm được và xác minh được bộ video camera thông thường có ground truth nghẹn ở trẻ đáp ứng
  mục tiêu này. Kết quả tìm kiếm không chứng minh một dataset phù hợp hoặc có quyền sử dụng.
- ChildSUn/Open Images chỉ hỗ trợ nhận dạng vật. 100DOH hỗ trợ tương tác tay–vật.
  ChokePoint là nhận dạng người qua cửa, không phải choking. Nghiên cứu nuốt bằng nội soi/X-quang
  có loại tín hiệu khác, không được dùng để chứng minh hiệu quả của camera phòng.
- Dữ liệu diễn lại của người lớn/ma-nơ-canh chỉ kiểm thử phần mềm và hành động mô phỏng. Không tạo
  nhãn “trẻ đang nghẹn thật” từ những cảnh đó. Không tạo tình huống nghẹn hoặc cho vật vào miệng để thử.
- Tuân thủ `docs/ethics.md`: không thu thập/xử lý video trẻ thật trong workflow repository hiện tại.
  Đánh giá lâm sàng đại diện cần đối tác và quy trình riêng phù hợp; chưa có các điều kiện đó.
- Báo cáo phải tách nguồn simulation/clinical, có bác sĩ/reviewer xác định nhãn, timestamp diễn biến,
  nguyên nhân đã xác minh/chưa rõ, tuổi nhóm, góc nhìn, tình trạng cảm biến và điều kiện ánh sáng.
- Không chuyển `level == DANGER` thành `choking_alarm`. Điểm về vật gần miệng không phải độ chính xác
  nhận diện nghẹn. Trước khi có model nghẹn thật, cột dự đoán nghẹn phải vắng/không khả dụng.
- Đo recall sự kiện toàn bộ kể cả lúc không quan sát được, false alarms trên thời gian âm tính đã
  xác minh, độ trễ từ onset tham chiếu, tỷ lệ thiếu thông tin và kết quả theo từng góc/nhóm tuổi.
  Nhãn chưa rõ không được tính là âm tính để làm đẹp specificity hoặc cảnh báo nhầm/giờ.

## 5. Công việc hiện tại và điều kiện hoàn thành

Đã khảo sát code, nguồn hướng dẫn y khoa và thiết kế tiêu chí dữ liệu. Bổ sung evaluator metadata
riêng để kiểm tra đầu ra nghẹn sau này; không đọc video trẻ và không làm thay đổi cảnh báo runtime.
Các kiểm thử tổng hợp của evaluator không chứng minh model nhận diện nghẹn.

Chưa có: dataset lâm sàng phù hợp, model thời gian/đa tín hiệu cho nghẹn, đối chiếu chuyên gia,
kiểm thử đại diện ở trẻ, chứng cứ bao phủ mọi góc và xác nhận nguyên nhân đồ chơi. Vì vậy không được
gọi tính năng là hoàn thành hoặc bật chẩn đoán nghẹn tự động trong sản phẩm hiện tại.

Thông tin cần từ nhóm: có sử dụng microphone không; có đối tác chuyên môn và ground truth lâm sàng
hay chưa. Những thông tin này quyết định lựa chọn model và thiết kế nghiên cứu, không được tự giả định.

### Quyết định người dùng ngày 12/10/2026

Chọn **hình ảnh + âm thanh trên thiết bị**. Hiện **chưa có chuyên gia hoặc dữ liệu lâm sàng**. Không
thu âm hay bật microphone trong lượt nghiên cứu này. Quyết định này chưa giải quyết khoảng trống nhãn
và kiểm chứng; chưa có bằng chứng để bật nhận diện nghẹn thật.

### Evaluator đã bổ sung

`scripts/evaluate_choking.py` nhận ba CSV metadata, không mở video/âm thanh:

- Predictions: `recording_id,subject_id,t,choking_alarm,choking_observation`. Đây phải là đầu ra
  model nghẹn độc lập, không đổi tên cột `level` của runtime. Observation là chất lượng tín hiệu
  cần cho đánh giá nghẹn, không tự sao chép cờ nhìn thấy miệng.
- Annotations: `recording_id,subject_id,start,end,diagnosis,cause,review_reference`. Diagnosis:
  `choking/non_choking/uncertain`; cause: `toy/food/other/unknown`; interval `[start,end)`.
- Recordings: `recording_id,duration_s,evidence_kind,age_group,viewpoint`. Evidence kind:
  `simulation/clinical`; recording là một phiên subject đã đồng bộ, không các ID tracker cục bộ
  từ các camera chưa được liên kết. Cùng một subject không được xuất hiện trùng timestamp.

Evaluator giữ cả người không có dự đoán trong denominator, không coi nhãn chưa rõ là âm tính,
tách điểm simulation/clinical và nhóm tuổi/góc nhìn. Một báo cáo điểm cao không tự chứng minh
ground truth là xác thực hoặc model có giá trị lâm sàng. Không xuất nhãn chẩn đoán từ evaluator.
