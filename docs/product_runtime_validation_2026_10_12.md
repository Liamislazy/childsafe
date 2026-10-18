# Kiểm chứng phần mềm nhánh sản phẩm — 12/10/2026

Nhánh `codex/product-runtime`, từ `97022bc`; không push/PR, train hoặc tải dataset
lớn. Đã triển khai phần phần mềm trong kế hoạch; chưa nghiệm thu trên cảnh người
lớn/ma-nơ-canh thật và chưa hiệu chuẩn camera. Không có tính năng chẩn đoán nghẹn.

## Các thay đổi

- Chất lượng ROI mặt dùng ảnh/tọa độ gốc, gate pixel không nhân hệ số zoom;
  log quality schema v3 cùng kích thước gốc. Sửa mismatch tọa độ gốc với crop người.
- Browser một phiên sở hữu, handshake clock và uncertainty, sequence/capture
  monotonic, lease/renew, reset floor; reject duplicate/stale/unowned requests.
  Không trực tiếp trừ hai clock browser/server. Snapshot timestamp không phải sensor.
- Deadline trước/sau inference cả web và CLI. Kết quả hết hạn không thêm nguy cơ
  mới; khôi phục processor trước frame hết hạn để giữ nguy cơ cũ ngắn, replay cùng
  lý do observation_lost. Reset khi đang xử lý trả busy, stop loại reply đang chạy.
- Hand crop/classifier và weak mask/box chỉ nghi ngờ; không thay thế mask định vị.
- Số liệu stage timing, processed gaps, slot overwrites, peak memory toàn process;
  hash model/config theo feature và API. CSV/manifest khác hash bị từ chối.
- Phiên adult/doll, consent khai báo rõ, subject tách dev/test, template observer,
  marker độc lập với inference. Đề xuất quality chỉ trên dev, chưa tự chứng nhận.
- Report theo condition, duration khai báo đầy đủ, người bỏ sót, unknown exposure,
  negative hours rõ nhãn, baseline/ablations và khoảng tin cậy có giả định.
- UI tiếng Việt, thiếu quan sát không màu xanh, model thật và giới hạn rõ.

## Bằng chứng đã có

Suite cuối: **163 passed, 3 skipped**, 17 cảnh báo deprecation, 14,41 giây.
Notebook Kaggle được đồng bộ lại với config JSON; không chạy train. Test model local thật:
1 passed trên crop đồ vật đã duyệt; không dùng ảnh/video người.

Trình duyệt Edge headless đã chạy frontend/API local với stream canvas hình học
640×480, bật/tắt phiên và ghi/xuất nhãn observer thành công, không lỗi JavaScript; không gọi camera/micro
thật. Screenshot được xem để kiểm tra bố cục và trạng thái xám. Máy chủ thử đã
dừng sau phép kiểm tra. Tải này không chứa người, không đo event recall.

Kiểm tra dependency: 79 package hiện có tương thích (`uv pip check`); chưa dựng
venv sạch khác và chưa đo GPU. Model/checkpoint hiện tại là CPU Torch 2.6.0+cpu.

Benchmark chạy 15 mẫu sau 3 warmup mỗi policy, real YOLO/classifier với fixture
người/mốc/mask tổng hợp, tám CPU threads; **không có MediaPipe thật trong timing**:

| Policy ROI | Median ms | p90 ms | Detector calls/frame |
|---|---:|---:|---:|
| Luôn chạy zoom | 268,50 | 278,66 | 2 |
| Bỏ zoom khi mask đã rõ | 132,35 | 151,49 | 1 |

Policy skip đã có từ `6cafaeb`; đây là đo lại trên code hiện tại, không tuyên bố
một cải thiện tốc độ mới trong đợt này. Artifact: `outputs/product_optimization_benchmark.json`.
Không suy ra FPS/latency 1/2/4 người từ fixture hoặc canvas không người.

## Chưa được chứng minh

Hiệu chuẩn quality/rule trên dev thật; recall/FA/giờ/latency sự kiện test độc lập;
góc cao và mọi góc; detector bút/đồng xu; GPU và clean install; chuyển giao sang trẻ.
Các metric này chưa đo, không thay bằng số tổng hợp. Calibration/coverage flags
vẫn false. Hướng dẫn thao tác và giới hạn ở `product_runtime_guide.md`.
