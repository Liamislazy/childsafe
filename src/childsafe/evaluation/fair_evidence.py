"""Readable evidence from the strict session evaluator; never invent pilot results."""
import csv
import json
from pathlib import Path

from childsafe.evaluation.product_report import METRIC_VERSION, pool_reports
from childsafe.evaluation.session_reports import build_report, load_records

FIELDS = ('events', 'hits', 'event_recall', 'false_alarms_per_hour',
          'latency_median_s', 'latency_p90_s', 'image_observation_coverage',
          'observation_coverage', 'negative_hours', 'duration_s')


def collect_evidence(root, split='test'):
    if split not in ('dev', 'test'):
        raise ValueError('select dev or test explicitly')
    root = Path(root)
    if not list(root.glob('*/manifest.json')):
        return dict(status='not_measured', split=split, metric_definition=METRIC_VERSION,
                    report=None, comparison=[],
                    note='Chưa có phiên ghi nhãn; không có recall, cảnh báo nhầm/giờ hoặc độ trễ camera.')
    # Invalid, unfinished or changed sessions must fail here, not become plausible numbers.
    report = build_report(load_records(root, split))
    comparisons = []
    for condition in sorted(report['by_condition']):
        sessions = [s for s in report['sessions'] if s['condition'] == condition]
        for policy, values in (
            ('full_danger', pool_reports(sessions)),
            ('distance_only', pool_reports([s['baseline'] for s in sessions])),
            ('full_caution_or_danger', pool_reports([s['caution_or_danger'] for s in sessions])),
        ):
            comparisons.append(dict(condition=condition, policy=policy,
                                    **{key:values.get(key) for key in FIELDS}))
    complete = bool(report['overall']['events'] and report['overall']['negative_hours'])
    return dict(status='sessions_with_positive_and_negative_labels' if complete else 'partial_labeled_sessions',
                split=split, metric_definition=METRIC_VERSION, report=report, comparison=comparisons,
                note='Chỉ kiểm tra nguồn gốc số liệu và nhãn đã khai báo; không xác thực cảnh quay hoặc hiệu quả ở trẻ.')


def _number(value, percent=False):
    if value is None:
        return 'chưa đo'
    return f'{100*value:.1f}%' if percent else f'{value:.3g}'


def write_evidence(bundle, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'evidence.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2,
                                                      allow_nan=False), encoding='utf8')
    with (destination/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=('condition', 'policy', *FIELDS))
        writer.writeheader()
        writer.writerows(bundle['comparison'])
    lines = ['# Minh chứng cảnh báo ChildSafe', '', f"Phân vùng dữ liệu: `{bundle['split']}`.",
             '', bundle['note'], '',
             'Phạm vi: mô phỏng vật tiến sát vùng miệng trên ảnh; không phải nghẹn, nuốt hoặc tiếp xúc vật lý.', '']
    if not bundle['comparison']:
        lines += ['**Chưa có dữ liệu kiểm thử phiên.**', '',
                  '| Chỉ số | Kết quả |', '|---|---|',
                  '| Recall toàn phiên | Chưa đo |', '| Cảnh báo nhầm mỗi giờ | Chưa đo |',
                  '| Độ trễ median/p90 | Chưa đo |', '| Tỷ lệ quan sát đủ rõ | Chưa đo |']
    else:
        lines += ['| Điều kiện | Chính sách | Sự kiện | Recall tổng | Báo nhầm/giờ | Trễ median/p90 (s) | Ảnh đủ rõ |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for row in bundle['comparison']:
            condition = str(row['condition']).replace('|', '/').replace('\n', ' ')
            lines.append(f"| {condition} | {row['policy']} | {row['events']} | "
                         f"{_number(row['event_recall'], True)} | {_number(row['false_alarms_per_hour'])} | "
                         f"{_number(row['latency_median_s'])}/{_number(row['latency_p90_s'])} | "
                         f"{_number(row['image_observation_coverage'], True)} |")
        lines += ['', 'Các mức CAUTION-or-DANGER và DANGER được báo riêng. Mất quan sát vẫn nằm trong recall tổng.',
                  'Khoảng tin cậy, số phiên, thời lượng âm tính và provenance nằm trong evidence.json.',
                  'So sánh trên cùng recall cần chọn ngưỡng bằng dev trước khi thu test; bảng này không tự bảo đảm hai chính sách có cùng recall.']
    lines += ['', 'Đây là minh chứng sơ bộ theo phiên đã khai báo, không phải chứng nhận an toàn hoặc hiệu quả triển khai.',
              'Độ trễ nhãn là thao tác người quan sát đến server; chưa đo độ trễ phản ứng của người gán nhãn.',
              'Không dùng số kiểm thử tự động hoặc benchmark ảnh tổng hợp thay cho những chỉ số trên.', '']
    (destination/'summary.md').write_text('\n'.join(lines), encoding='utf8')
    return destination
