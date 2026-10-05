"""Run beside the existing app.py on the Pi to serve web/index.html."""
from datetime import datetime
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
app = root / "app.py"
page = root / "web" / "index.html"
marker = "# Topgun upload page"
if not page.is_file():
    raise SystemExit("Missing web/index.html")
source = app.read_text()
if marker not in source:
    addition = '''

# Topgun upload page
from pathlib import Path as _WebPath
from fastapi.responses import FileResponse as _WebFileResponse

@app.get("/", include_in_schema=False)
def topgun_upload_page():
    return _WebFileResponse(_WebPath(__file__).resolve().parent / "web" / "index.html")
'''
    compile(source + addition, str(app), "exec")
    shutil.copy2(app, app.with_name("app.py.backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    app.write_text(source + addition)
    print("Web page route added; restart uvicorn.")
else:
    print("Web page route already exists.")
