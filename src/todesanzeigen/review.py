from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm import CSV_COLUMNS
from .storage import (
    DEFAULT_DB_PATH,
    DEFAULT_LABEL_SET,
    apply_migrations,
    connect,
    document_review_detail,
    fields_from_form,
    load_candidate,
    mark_candidate_status,
    pending_review_items,
    save_ground_truth_label,
)


def create_review_app(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    label_set: str = DEFAULT_LABEL_SET,
    reviewer: str = "",
) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
        from jinja2 import Template
    except ImportError as exc:
        raise RuntimeError(
            "Review UI requires fastapi, uvicorn, and jinja2. "
            "Install project dependencies before running `todesanzeigen review serve`."
        ) from exc

    apply_migrations(db_path)
    app = FastAPI(title="Todesanzeigen Review")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        with connect(db_path) as connection:
            items = [dict(row) for row in pending_review_items(connection, label_set=label_set)]
        return HTMLResponse(
            Template(INDEX_TEMPLATE).render(items=items, label_set=label_set)
        )

    @app.get("/documents/{document_id}", response_class=HTMLResponse)
    def document(document_id: int) -> HTMLResponse:
        with connect(db_path) as connection:
            detail = document_review_detail(
                connection,
                document_id=document_id,
                label_set=label_set,
            )
        fields = _initial_fields(detail)
        return HTMLResponse(
            Template(DOCUMENT_TEMPLATE).render(
                detail=detail,
                fields=fields,
                columns=CSV_COLUMNS,
                label_set=label_set,
            )
        )

    @app.get("/images/{document_id}")
    def image(document_id: int) -> Any:
        with connect(db_path) as connection:
            detail = document_review_detail(connection, document_id=document_id, label_set=label_set)
        image_path = Path(detail["document"].get("image_path", ""))
        if not image_path.exists():
            return HTMLResponse("Image not found", status_code=404)
        return FileResponse(image_path)

    @app.post("/candidates/{candidate_id}/approve")
    async def approve_candidate(candidate_id: int) -> RedirectResponse:
        with connect(db_path) as connection:
            candidate = load_candidate(connection, candidate_id)
            save_ground_truth_label(
                connection,
                document_id=int(candidate["document_id"]),
                label_set=label_set,
                fields=_loads(candidate["fields_json"]),
                source_candidate_id=candidate_id,
                reviewer=reviewer,
            )
            document_id = int(candidate["document_id"])
        return RedirectResponse(f"/documents/{document_id}", status_code=303)

    @app.post("/candidates/{candidate_id}/needs-review")
    async def needs_review(candidate_id: int) -> RedirectResponse:
        with connect(db_path) as connection:
            candidate = load_candidate(connection, candidate_id)
            mark_candidate_status(connection, candidate_id=candidate_id, status="needs_review")
            document_id = int(candidate["document_id"])
        return RedirectResponse(f"/documents/{document_id}", status_code=303)

    @app.post("/documents/{document_id}/labels")
    async def save_label(document_id: int, request: Request) -> RedirectResponse:
        form = await request.form()
        source_candidate_id = _optional_int(form.get("source_candidate_id"))
        review_notes = str(form.get("review_notes", "") or "")
        with connect(db_path) as connection:
            save_ground_truth_label(
                connection,
                document_id=document_id,
                label_set=label_set,
                fields=fields_from_form(form.multi_items()),
                source_candidate_id=source_candidate_id,
                reviewer=reviewer,
                review_notes=review_notes,
            )
        return RedirectResponse(f"/documents/{document_id}", status_code=303)

    return app


def serve_review_app(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    label_set: str = DEFAULT_LABEL_SET,
    reviewer: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Review UI requires uvicorn. Install project dependencies before running it."
        ) from exc
    app = create_review_app(db_path=db_path, label_set=label_set, reviewer=reviewer)
    uvicorn.run(app, host=host, port=port)


def _initial_fields(detail: dict[str, Any]) -> dict[str, str]:
    if detail["ground_truth"] is not None:
        return {column: str(detail["ground_truth"]["fields"].get(column, "")) for column in CSV_COLUMNS}
    if detail["candidates"]:
        return {column: str(detail["candidates"][0]["fields"].get(column, "")) for column in CSV_COLUMNS}
    return {column: "" for column in CSV_COLUMNS}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _loads(value: str) -> dict[str, Any]:
    import json

    data = json.loads(value or "{}")
    return data if isinstance(data, dict) else {}


INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Todesanzeigen Review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; color: #1f2933; background: #f7f7f4; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    table { width: 100%; border-collapse: collapse; background: white; }
    th, td { border-bottom: 1px solid #dde1e4; padding: 10px; text-align: left; }
    a { color: #174c7c; }
  </style>
</head>
<body>
<main>
  <h1>Review Queue: {{ label_set }}</h1>
  <table>
    <thead><tr><th>Document</th><th>Source</th><th>Candidate</th><th>Image</th></tr></thead>
    <tbody>
    {% for item in items %}
      <tr>
        <td><a href="/documents/{{ item.document_id }}">{{ item.filename_stem }}</a></td>
        <td>{{ item.source }}</td>
        <td>{{ item.source_kind }} {{ item.source_name }}</td>
        <td>{{ item.image_path }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</main>
</body>
</html>
"""


DOCUMENT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ detail.document.filename_stem }}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; color: #1f2933; background: #f7f7f4; }
    header { padding: 16px 24px; background: #263238; color: white; }
    main { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(360px, 1fr); gap: 20px; padding: 20px; }
    img { max-width: 100%; background: white; border: 1px solid #d9dee2; }
    textarea { width: 100%; min-height: 120px; font-family: ui-monospace, monospace; }
    label { display: block; font-size: 13px; margin: 8px 0 4px; }
    input, textarea { box-sizing: border-box; width: 100%; padding: 8px; border: 1px solid #b8c0c7; border-radius: 4px; }
    button { padding: 8px 12px; border: 0; border-radius: 4px; background: #174c7c; color: white; cursor: pointer; }
    .panel { background: white; border: 1px solid #d9dee2; padding: 16px; }
    .candidate { border-top: 1px solid #d9dee2; padding-top: 12px; margin-top: 12px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <a href="/" style="color:white">Queue</a>
  <h1>{{ detail.document.filename_stem }}</h1>
  <div>{{ detail.document.source }} | {{ label_set }}</div>
</header>
<main>
  <section>
    <img src="/images/{{ detail.document.id }}" alt="{{ detail.document.filename_stem }}">
    <div class="panel">
      <h2>OCR Text</h2>
      <textarea readonly>{{ detail.ocr.text if detail.ocr else "" }}</textarea>
    </div>
  </section>
  <section class="panel">
    <h2>Ground Truth Editor</h2>
    <form method="post" action="/documents/{{ detail.document.id }}/labels">
      <input type="hidden" name="source_candidate_id" value="{{ detail.candidates[0].id if detail.candidates else '' }}">
      <div class="grid">
      {% for column in columns %}
        <div>
          <label for="{{ column }}">{{ column }}</label>
          <input id="{{ column }}" name="{{ column }}" value="{{ fields[column] }}">
        </div>
      {% endfor %}
      </div>
      <label for="review_notes">review_notes</label>
      <textarea id="review_notes" name="review_notes">{{ detail.ground_truth.review_notes if detail.ground_truth else "" }}</textarea>
      <button type="submit">Save Reviewed Label</button>
    </form>
    <h2>Candidates</h2>
    {% for candidate in detail.candidates %}
      <div class="candidate">
        <div>{{ candidate.source_kind }} {{ candidate.source_name }} | {{ candidate.status }}</div>
        <form method="post" action="/candidates/{{ candidate.id }}/approve">
          <button type="submit">Approve As Ground Truth</button>
        </form>
        <form method="post" action="/candidates/{{ candidate.id }}/needs-review">
          <button type="submit">Mark Needs Review</button>
        </form>
      </div>
    {% endfor %}
  </section>
</main>
</body>
</html>
"""
