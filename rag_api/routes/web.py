from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"

# Everything is served from this origin: no CDN, no inline scripts or styles.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)

router = APIRouter(include_in_schema=False)


@router.get("/ui")
def web_ui() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Content-Security-Policy": CONTENT_SECURITY_POLICY,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-cache",
        },
    )
