# Pages and dependencies
/ -> src/childsafe/web/static/index.html
  -> inline CSS
  -> inline vanilla JavaScript
  -> browser camera and canvas APIs
  -> /api/info, /api/frame, /api/camera/*, /api/session.*, /api/annotation(s.csv)
No local imports, assets, bundled icons, fonts, framework config or external style dependencies.
New Android UI must use native controls and retain the risk/observation distinction.
