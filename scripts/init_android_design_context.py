"""Record the existing web UI as context for the new native Android target."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / ".superdesign" / "init"
folder.mkdir(parents=True, exist_ok=True)
source = (root / "src/childsafe/web/static/index.html").read_text(encoding="utf-8")
style = source.split("<style>", 1)[1].split("</style>", 1)[0]
files = {
    "components.md": "# Components\nSingle vanilla HTML page; no imports, component library or existing native Android UI.\n"
        "The following complete source contains all inline components and conditional render branches.\n```html\n" + source + "\n```\n",
    "layouts.md": "# Layouts\nNo separate shell files. Header, scope disclosure, preview and status columns live in this complete source.\n```html\n" + source + "\n```\n",
    "routes.md": "# Routes\nGET / -> src/childsafe/web/static/index.html.\n"
        "POST /api/frame -> local Python inference, not a native Android screen.\n"
        "New target: Android monitor dashboard and camera/monitor role selection; not yet implemented.\n",
    "theme.md": "# Theme\nSystem sans serif, light/dark themes; blue primary action, amber caution, red danger.\n"
        "Normal with insufficient observation must use neutral gray, not green.\n"
        "All existing CSS tokens and selectors:\n```css\n" + style + "\n```\n",
    "pages.md": "# Pages and dependencies\n/ -> src/childsafe/web/static/index.html\n"
        "  -> inline CSS\n  -> inline vanilla JavaScript\n  -> browser camera and canvas APIs\n"
        "  -> /api/info, /api/frame, /api/camera/*, /api/session.*, /api/annotation(s.csv)\n"
        "No local imports, assets, bundled icons, fonts, framework config or external style dependencies.\n"
        "New Android UI must use native controls and retain the risk/observation distinction.\n",
    "extractable-components.md": "# Extractable components\n"
        "No reusable framework or native layout components exist. The web header and cards are inline HTML, "
        "not shared Android components. Skip component extraction for this native target.\n",
}
for name, content in files.items():
    (folder / name).write_text(content, encoding="utf-8")
print("Wrote six nonempty init files.")
