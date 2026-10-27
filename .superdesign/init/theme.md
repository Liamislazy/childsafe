# Theme
System sans serif, light/dark themes; blue primary action, amber caution, red danger.
Normal with insufficient observation must use neutral gray, not green.
All existing CSS tokens and selectors:
```css

  :root {
    --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #667085; --line: #e3e6eb;
    --normal: #1f9d55; --caution: #d98a00; --danger: #d33a2c; --accent: #3b6fd8; --chip: #eef1f6;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #12151b; --panel: #1b2029; --text: #e7eaf0; --muted: #98a2b3; --line: #2b3240;
            --normal: #34c374; --caution: #f0a92b; --danger: #ef5b4c; --accent: #6d97f0; --chip: #252c38; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--line); display: flex; gap: 16px;
           align-items: center; flex-wrap: wrap; background: var(--panel); }
  header h1 { font-size: 17px; margin: 0; }
  .privacy { color: var(--muted); font-size: 12px; }
  .tabs { margin-left: auto; display: flex; gap: 4px; }
  .tabs button { border: 1px solid var(--line); background: var(--panel); color: var(--text);
                 padding: 6px 14px; border-radius: 8px; cursor: pointer; }
  .tabs button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  main { padding: 16px 20px; max-width: 1400px; margin: 0 auto; }
  .grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 16px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .stage { position: relative; width: 100%; background: #000; border-radius: 10px; overflow: hidden;
           aspect-ratio: 4 / 3; }
  .stage video, .stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
  .stage video { transform: scaleX(-1); }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; margin-top: 12px; }
  .controls label { color: var(--muted); font-size: 12px; display: flex; flex-direction: column; gap: 3px; }
  select, input[type=range] { accent-color: var(--accent); }
  select { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 4px 6px; }
  .btn { border: 0; border-radius: 8px; padding: 8px 14px; cursor: pointer; font-weight: 600;
         background: var(--accent); color: #fff; }
  .btn.ghost { background: var(--chip); color: var(--text); }
  .stats { color: var(--muted); font-size: 12px; margin-left: auto; }
  .models { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; align-content: start; }
  @media (max-width: 520px) { .models { grid-template-columns: 1fr; } }
  .model h3 { margin: 0 0 2px; font-size: 14px; }
  .model .sub { color: var(--muted); font-size: 11px; margin-bottom: 8px; word-break: break-all; }
  .level { border-radius: 8px; padding: 10px; color: #fff; font-weight: 700; font-size: 18px;
           text-align: center; letter-spacing: .04em; background: var(--muted); }
  .level.normal { background: var(--normal); } .level.caution { background: var(--caution); }
  .level.danger { background: var(--danger); }
  .level.limited { background: #6b7280; }
  .reason { min-height: 1.4em; color: var(--muted); font-size: 12px; text-align: center; margin: 4px 0 8px; }
  .bar { height: 10px; background: var(--chip); border-radius: 6px; overflow: hidden; margin: 4px 0 2px; }
  .bar > div { height: 100%; width: 0; background: var(--danger); transition: width .15s; }
  .row { display: flex; justify-content: space-between; font-size: 12px; }
  .row span:first-child { color: var(--muted); }
  ol.top { margin: 8px 0 0; padding-left: 18px; font-size: 13px; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; background: var(--chip); }
  .badge.unsafe { background: var(--danger); color: #fff; } .badge.safe { background: var(--normal); color: #fff; }
  .badge.uncertain { background: var(--caution); color: #fff; }
  .seen { display: flex; gap: 12px; align-items: center; margin-top: 12px; }
  .seen img { width: 96px; height: 96px; object-fit: cover; border-radius: 8px; border: 1px solid var(--line); background: var(--chip); }
  .note { color: var(--muted); font-size: 12px; margin-top: 10px; }
  .hidden { display: none; }
  .err { color: var(--danger); font-size: 13px; margin-top: 8px; }

```
