# ChildSafe Android companion

Product: two roles in one Android app. A spare phone is the camera, the main phone
receives event notifications. Camera inference stays on the device. A Windows
webcam node can use the same encrypted event protocol. Internet relay deployment
and Android push setup must be shown as configuration pending until tested.

Audience: Vietnamese parents/caregivers. Native mobile controls, minimum 48dp
touch targets and readable 16sp body text. System Roboto/sans serif only.
No photographs of children, invented product logos or diagnostic claims.

Visual system: use the existing web application's light theme. Background #f6f7f9,
cards #ffffff, text #1d2330, secondary #667085, dividers #e3e6eb, primary #3b6fd8,
caution #d98a00, danger #d33a2c. Dark equivalents from init/theme.md. Neutral gray
for unavailable observation/connection. Green #1f9d55 only for fresh, observable
normal results. 8dp spacing unit, 12dp card radius, no gradients or decorative art.

Primary dashboard: connection and observation status first; camera device cards;
text and icon together for risk; prominent 'Kiểm tra ngay' for high risk; event
list; mute/acknowledge applies to notification sound, not to current risk.
Toolbar 'ChildSafe', a secondary subtitle 'Hỗ trợ người chăm sóc'. Distinguish
'Mất kết nối', 'Chưa quan sát rõ', 'Chưa ghi nhận nguy cơ', 'Cần chú ý', 'Nguy cơ cao'.
Never say 'Trẻ an toàn', 'Đã phát hiện nghẹn', or promise prevention.

Pairing flow: choose 'Máy theo dõi' or 'Máy camera'; pair by expiring one-use QR;
permission prompts explain camera and notification purpose. Camera preview is
local only. Default relay payload contains encrypted event data, no video or audio.
No baby images or stock imagery; simple native Material-style outline icons.

Motion: short state transitions only. No pulsing status suggesting life monitoring.
This is a research prototype; show a small persistent 'Nguyên mẫu đang kiểm chứng'
label, not a modal that blocks every interaction.
