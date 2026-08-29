from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

try:
    from fastapi import Request as FastApiRequest
except ImportError:
    FastApiRequest = Any

from .llm import STORED_COLUMNS
from .storage import (
    DEFAULT_DB_PATH,
    DEFAULT_LABEL_SET,
    apply_migrations,
    connect,
    document_review_detail,
    fields_from_form,
    load_candidate,
    next_pending_review_document,
    pending_review_count,
    pending_review_items,
    review_method_options,
    reviewed_ground_truth_count,
    reviewed_ground_truth_items,
    save_ground_truth_label,
)

PAGE_SIZE = 50
FIELD_LABELS = {
    "geschlecht": "Geschlecht", "name": "Nachname", "vorname": "Vorname",
    "foto": "Foto", "geburtsdatum": "Geburtsdatum", "sterbedatum": "Sterbedatum",
    "geburtsname": "Geburtsname", "titel": "Titel", "genannt": "Genannt",
    "geburtsort": "Geburtsort", "sterbeort": "Sterbeort", "ort": "Wohnort / Ort",
    "weitere_orte": "Weitere Orte", "beruf": "Beruf", "bemerkungen": "Bemerkungen",
    "quelle": "Quelle", "dateiname": "Dateiname",
    "zusaetzliche_hinweise": "Zusätzliche Hinweise", "confidence_score": "Konfidenz",
}
FIELD_GROUPS = (
    ("Person", ("geschlecht", "titel", "vorname", "name", "geburtsname", "genannt", "foto")),
    ("Lebensdaten", ("geburtsdatum", "sterbedatum", "geburtsort", "sterbeort", "ort", "weitere_orte")),
    ("Weitere Angaben", ("beruf", "bemerkungen", "zusaetzliche_hinweise")),
    ("Metadaten", ("quelle", "dateiname", "confidence_score")),
)
LONG_FIELDS = {"weitere_orte", "beruf", "bemerkungen", "zusaetzliche_hinweise"}


def create_review_app(
    *, db_path: Path = DEFAULT_DB_PATH, label_set: str = DEFAULT_LABEL_SET, reviewer: str = ""
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
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
    def index(view: str = "pending", method: str = "", page: int = 1) -> HTMLResponse:
        current_view = "reviewed" if view == "reviewed" else "pending"
        method_filter = method.strip() if current_view == "pending" else ""
        with connect(db_path) as connection:
            pending_total = pending_review_count(connection, label_set=label_set)
            filtered_pending_count = pending_review_count(
                connection, label_set=label_set, source_name=method_filter
            )
            reviewed_count = reviewed_ground_truth_count(connection, label_set=label_set)
            total_count = filtered_pending_count if current_view == "pending" else reviewed_count
            page_count = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
            current_page = min(max(page, 1), page_count)
            offset = (current_page - 1) * PAGE_SIZE
            if current_view == "pending":
                items = [
                    _prepare_pending_item(dict(row))
                    for row in pending_review_items(
                        connection, label_set=label_set, source_name=method_filter,
                        limit=PAGE_SIZE, offset=offset,
                    )
                ]
            else:
                items = [
                    _prepare_reviewed_item(dict(row))
                    for row in reviewed_ground_truth_items(
                        connection, label_set=label_set, limit=PAGE_SIZE, offset=offset
                    )
                ]
            method_options = [dict(row) for row in review_method_options(connection)]
        return HTMLResponse(Template(INDEX_TEMPLATE).render(
            items=items, label_set=label_set, method_filter=method_filter,
            method_options=method_options, view=current_view, pending_count=pending_total,
            filtered_pending_count=filtered_pending_count, reviewed_count=reviewed_count,
            page=current_page, page_count=page_count,
        ))

    @app.get("/documents/{document_id}", response_class=HTMLResponse)
    def document(document_id: int, method: str = "", view: str = "pending", page: int = 1) -> HTMLResponse:
        with connect(db_path) as connection:
            detail = document_review_detail(connection, document_id=document_id, label_set=label_set)
        fields = _initial_fields(detail)
        candidates = [_prepare_candidate(candidate) for candidate in detail["candidates"]]
        candidates.sort(key=_candidate_sort_key)
        detail["candidates"] = candidates
        comparison_sources: list[dict[str, Any]] = []
        if detail["ground_truth"] is not None:
            comparison_sources.append({
                "kind": "ground_truth", "label": "Current ground truth",
                "fields": _fields(detail["ground_truth"]["fields"]),
            })
        comparison_sources.extend(candidates)
        different_fields = {
            column for column in STORED_COLUMNS
            if len({source["fields"].get(column, "") for source in comparison_sources}) > 1
        }
        return HTMLResponse(Template(DOCUMENT_TEMPLATE).render(
            detail=detail, fields=fields, field_groups=FIELD_GROUPS, field_labels=FIELD_LABELS,
            long_fields=LONG_FIELDS, columns=STORED_COLUMNS, comparison_sources=comparison_sources,
            different_fields=different_fields,
            candidate_payload=[{"id": c["id"], "label": c["label"], "fields": c["fields"]} for c in candidates],
            label_set=label_set, source_candidate_id=_source_candidate_id(detail),
            initial_source_label=_initial_source_label(detail), method_filter=method.strip(),
            return_view="reviewed" if view == "reviewed" else "pending", return_page=max(page, 1),
        ))

    @app.get("/images/{document_id}")
    def image(document_id: int) -> Any:
        with connect(db_path) as connection:
            detail = document_review_detail(connection, document_id=document_id, label_set=label_set)
        image_path = Path(detail["document"].get("image_path", ""))
        if not image_path.exists():
            return HTMLResponse("Image not found", status_code=404)
        return FileResponse(image_path)

    @app.post("/candidates/{candidate_id}/approve")
    async def approve_candidate(candidate_id: int, request: FastApiRequest) -> RedirectResponse:
        form = await request.form()
        method_filter = str(form.get("method", "") or "").strip()
        with connect(db_path) as connection:
            candidate = load_candidate(connection, candidate_id)
            document_id = int(candidate["document_id"])
            existing = connection.execute(
                "SELECT review_notes FROM ground_truth_labels WHERE document_id = ? AND label_set = ?",
                (document_id, label_set),
            ).fetchone()
            save_ground_truth_label(
                connection, document_id=document_id, label_set=label_set,
                fields=_loads(candidate["fields_json"]), source_candidate_id=candidate_id,
                reviewer=reviewer,
                review_notes=str(existing["review_notes"]) if existing is not None else "",
            )
            target = _continuation_target(
                connection, document_id=document_id, label_set=label_set,
                method_filter=method_filter, continue_mode=_continue_mode(form.get("continue_mode")),
            )
        return RedirectResponse(target, status_code=303)

    @app.post("/documents/{document_id}/needs-review")
    async def needs_review(document_id: int, request: FastApiRequest) -> RedirectResponse:
        form = await request.form()
        with connect(db_path) as connection:
            target = _continuation_target(
                connection, document_id=document_id, label_set=label_set,
                method_filter=str(form.get("method", "") or "").strip(), continue_mode="next",
            )
        return RedirectResponse(target, status_code=303)

    @app.post("/documents/{document_id}/labels")
    async def save_label(document_id: int, request: FastApiRequest) -> RedirectResponse:
        form = await request.form()
        method_filter = str(form.get("method", "") or "").strip()
        with connect(db_path) as connection:
            try:
                save_ground_truth_label(
                    connection, document_id=document_id, label_set=label_set,
                    fields=fields_from_form(form.multi_items()),
                    source_candidate_id=_optional_int(form.get("source_candidate_id")),
                    reviewer=reviewer, review_notes=str(form.get("review_notes", "") or ""),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            target = _continuation_target(
                connection, document_id=document_id, label_set=label_set,
                method_filter=method_filter, continue_mode=_continue_mode(form.get("continue_mode")),
            )
        return RedirectResponse(target, status_code=303)

    return app


def serve_review_app(
    *, db_path: Path = DEFAULT_DB_PATH, label_set: str = DEFAULT_LABEL_SET,
    reviewer: str = "", host: str = "127.0.0.1", port: int = 8000,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Review UI requires uvicorn. Install project dependencies before running it.") from exc
    uvicorn.run(create_review_app(db_path=db_path, label_set=label_set, reviewer=reviewer), host=host, port=port)


def _initial_fields(detail: dict[str, Any]) -> dict[str, str]:
    return _fields(detail["ground_truth"]["fields"]) if detail["ground_truth"] is not None else _fields({})


def _source_candidate_id(detail: dict[str, Any]) -> str:
    if detail["ground_truth"] is None:
        return ""
    value = detail["ground_truth"].get("source_candidate_id")
    return "" if value in (None, "") else str(value)


def _initial_source_label(detail: dict[str, Any]) -> str:
    ground_truth = detail["ground_truth"]
    if ground_truth is None:
        return "No source selected"
    source = ground_truth.get("extraction_method") or ground_truth.get("source_name") or "manual"
    return f"Current GT · {_method_label(str(source))}"


def _prepare_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    method = str(candidate.get("extraction_method") or candidate.get("source_name") or "candidate")
    prepared = {**candidate, "fields": _fields(candidate.get("fields", {})), "label": _method_label(method)}
    prepared["meta"] = " · ".join(
        value for value in (
            str(candidate.get("method_family") or ""), str(candidate.get("extraction_provider") or ""),
            str(candidate.get("extraction_model") or ""), str(candidate.get("route_reason") or ""),
        ) if value
    )
    return prepared


def _prepare_pending_item(item: dict[str, Any]) -> dict[str, Any]:
    methods = [value for value in str(item.get("candidate_methods") or "").split(",") if value]
    priorities = {"text_extraction": 0, "vision_model_image_only": 1, "vision_model_reroute": 2}
    methods.sort(key=lambda method: (priorities.get(method, 99), method))
    return {**item, "methods": [_method_label(method) for method in methods]}


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, str]:
    method = str(candidate.get("extraction_method") or candidate.get("source_name") or "")
    priority = {
        "text_extraction": 0,
        "vision_model_image_only": 1,
        "vision_model_reroute": 2,
    }
    return priority.get(method, 99), method


def _prepare_reviewed_item(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "gt_source_label": _method_label(str(item.get("source_name") or "manual"))}


def _method_label(method: str) -> str:
    return {
        "text_extraction": "Text extraction", "vision_model_image_only": "VLM · image only",
        "vision_model_reroute": "VLM · reroute", "manual": "Manual",
    }.get(method, method.replace("_", " ").strip().title())


def _fields(values: dict[str, Any]) -> dict[str, str]:
    return {column: str(values.get(column, "") or "") for column in STORED_COLUMNS}


def _continue_mode(value: Any) -> str:
    return "stay" if str(value or "").strip() == "stay" else "next"


def _continuation_target(
    connection: Any, *, document_id: int, label_set: str, method_filter: str, continue_mode: str
) -> str:
    method_query = f"?method={quote(method_filter)}" if method_filter else ""
    if continue_mode == "stay":
        return f"/documents/{document_id}{method_query}"
    next_id = next_pending_review_document(
        connection, current_document_id=document_id, label_set=label_set, source_name=method_filter
    )
    return f"/documents/{next_id}{method_query}" if next_id is not None else f"/{method_query}"


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
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ground Truth Review</title><style>
:root{--ink:#18232d;--muted:#62717e;--line:#d9e0e5;--brand:#175b75;--paper:#fff;--wash:#f3f6f5;--pending:#9a5b05;--pending-bg:#fff4db;--ok:#226342;--ok-bg:#e7f5ec}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--wash);font-family:Inter,system-ui,sans-serif}a{color:var(--brand)}main{width:min(1180px,calc(100% - 32px));margin:auto;padding:32px 0 56px}.eyebrow{margin:0 0 6px;color:var(--brand);font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}h1{margin:0;font-size:clamp(26px,4vw,38px);letter-spacing:-.025em}.subtitle{margin:8px 0 24px;color:var(--muted)}.tabs{display:flex;gap:8px;margin-bottom:18px;border-bottom:1px solid var(--line)}.tab{display:inline-flex;gap:8px;align-items:center;padding:12px 16px;margin-bottom:-1px;border:1px solid transparent;border-radius:8px 8px 0 0;color:var(--muted);text-decoration:none;font-weight:700}.tab.active{background:var(--paper);border-color:var(--line);border-bottom-color:var(--paper);color:var(--ink)}.count{min-width:24px;padding:2px 7px;border-radius:999px;background:#e8eef1;text-align:center;font-size:12px}.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}.filter-option{padding:7px 11px;border:1px solid #b8c5cc;border-radius:999px;background:#fff;text-decoration:none;font-size:13px;font-weight:650}.filter-option.active{border-color:var(--brand);background:var(--brand);color:#fff}.card{overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#fff;box-shadow:0 8px 24px #19303c0d}table{width:100%;border-collapse:collapse}th,td{padding:14px 16px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{background:#f8faf9;color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fbfdfc}.document-link{color:var(--ink);font-weight:750;text-decoration:none}.document-link:hover{color:var(--brand);text-decoration:underline}.subtle{margin-top:3px;color:var(--muted);font-size:12px}.badge,.status{display:inline-block;border-radius:999px;font-size:12px;font-weight:750}.badge{margin:2px 4px 2px 0;padding:4px 8px;background:#edf2f4;color:#38515e}.status{padding:5px 9px}.status.pending{color:var(--pending);background:var(--pending-bg)}.status.saved{color:var(--ok);background:var(--ok-bg)}.newer{margin-top:5px;color:#a34b15;font-size:12px;font-weight:700}.empty{padding:46px 20px;color:var(--muted);text-align:center}.pagination{display:flex;justify-content:space-between;align-items:center;margin-top:14px;color:var(--muted);font-size:13px}.page-link{padding:7px 11px;border:1px solid var(--line);border-radius:6px;background:#fff;text-decoration:none}@media(max-width:760px){main{width:calc(100% - 20px);padding-top:20px}.card{overflow-x:auto}th,td{min-width:150px}}
</style></head><body><main>
<p class="eyebrow">{{ label_set }}</p><h1>Ground Truth Review</h1><p class="subtitle">Choose a source, verify it against the notice, and save one authoritative label.</p>
<nav class="tabs"><a class="tab{% if view == 'pending' %} active{% endif %}" href="/?view=pending">Needs review <span class="count">{{ pending_count }}</span></a><a class="tab{% if view == 'reviewed' %} active{% endif %}" href="/?view=reviewed">Ground truth <span class="count">{{ reviewed_count }}</span></a></nav>
{% if view == 'pending' %}<nav class="filters"><a class="filter-option{% if not method_filter %} active{% endif %}" href="/?view=pending">All methods</a>{% for option in method_options %}<a class="filter-option{% if option.method == method_filter %} active{% endif %}" href="/?view=pending&amp;method={{ option.method|urlencode }}" title="{{ option.description }}">{{ option.method|replace('_',' ') }}</a>{% endfor %}</nav>{% endif %}
<section class="card"><table><thead>{% if view == 'pending' %}<tr><th>Document</th><th>Status</th><th>Available sources</th><th>Candidates</th></tr>{% else %}<tr><th>Document</th><th>Status</th><th>GT source</th><th>Reviewer / updated</th></tr>{% endif %}</thead><tbody>
{% for item in items %}<tr><td><a class="document-link" href="/documents/{{ item.document_id }}?view={{ view }}&amp;page={{ page }}{% if method_filter %}&amp;method={{ method_filter|urlencode }}{% endif %}">{{ item.filename_stem }}</a><div class="subtle">{{ item.source }}</div></td>
{% if view == 'pending' %}<td><span class="status pending">Needs review</span></td><td>{% for method in item.methods %}<span class="badge">{{ method }}</span>{% endfor %}</td><td>{{ item.candidate_count }}</td>
{% else %}<td><span class="status saved">GT available</span>{% if item.newer_candidate_count %}<div class="newer">{{ item.newer_candidate_count }} newer candidate{% if item.newer_candidate_count != 1 %}s{% endif %}</div>{% endif %}</td><td><span class="badge">{{ item.gt_source_label }}</span></td><td>{{ item.reviewer or '—' }}<div class="subtle">{{ item.updated_at }}</div></td>{% endif %}</tr>
{% else %}<tr><td class="empty" colspan="4">{% if view == 'pending' %}No records need ground truth for this filter.{% else %}No ground-truth labels have been saved yet.{% endif %}</td></tr>{% endfor %}</tbody></table></section>
{% if page_count > 1 %}<nav class="pagination">{% if page > 1 %}<a class="page-link" href="/?view={{ view }}&amp;page={{ page-1 }}{% if method_filter %}&amp;method={{ method_filter|urlencode }}{% endif %}">Previous</a>{% else %}<span></span>{% endif %}<span>Page {{ page }} of {{ page_count }}</span>{% if page < page_count %}<a class="page-link" href="/?view={{ view }}&amp;page={{ page+1 }}{% if method_filter %}&amp;method={{ method_filter|urlencode }}{% endif %}">Next</a>{% else %}<span></span>{% endif %}</nav>{% endif %}
</main></body></html>"""


DOCUMENT_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ detail.document.filename_stem }} · GT Review</title><style>
:root{--ink:#18232d;--muted:#647480;--line:#d9e0e5;--brand:#175b75;--brand-dark:#11485e;--paper:#fff;--wash:#f2f5f4;--pending:#9a5b05;--pending-bg:#fff4db;--ok:#226342;--ok-bg:#e7f5ec;--diff:#fff8df;--danger:#8a3e2c}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--wash);font-family:Inter,system-ui,sans-serif}a{color:var(--brand)}header{position:sticky;top:0;z-index:20;display:flex;justify-content:space-between;gap:20px;align-items:center;padding:13px 22px;background:#172b35;color:#fff;box-shadow:0 2px 10px #0002}header a{color:#d8eff7;text-decoration:none}.header-title{min-width:0}.header-title strong{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.header-title span{color:#b9cbd3;font-size:12px}.state{flex:none;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:800}.state.pending{color:#ffe5ad;background:#9a5b0547}.state.saved{color:#c7f2d8;background:#2263426b}main{display:grid;grid-template-columns:minmax(300px,38%) minmax(540px,1fr);gap:20px;width:min(1500px,calc(100% - 32px));margin:auto;padding:20px 0 56px}.evidence{min-width:0}.evidence-sticky{position:sticky;top:84px;display:grid;gap:14px}.panel{border:1px solid var(--line);border-radius:12px;background:#fff;box-shadow:0 8px 24px #19303c0d}.panel-pad{padding:18px}.notice-image{display:block;width:100%;max-height:calc(100vh - 120px);border-radius:12px;object-fit:contain;background:#e5e9e8}h2{margin:0;font-size:20px;letter-spacing:-.015em}.section-heading{display:flex;justify-content:space-between;gap:12px;align-items:start;margin-bottom:13px}.section-heading p,.help{margin:5px 0 0;color:var(--muted);font-size:13px}details summary{cursor:pointer;font-weight:750}.ocr-text{width:100%;max-height:260px;margin-top:12px;padding:12px;overflow:auto;border-radius:7px;background:#f6f8f7;color:#344650;white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace}.workspace{min-width:0;display:grid;gap:16px;align-content:start}.gt-summary{display:flex;justify-content:space-between;gap:16px;align-items:start}.status-badge{display:inline-block;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:800}.status-badge.pending{color:var(--pending);background:var(--pending-bg)}.status-badge.saved{color:var(--ok);background:var(--ok-bg)}.gt-meta{margin-top:8px;color:var(--muted);font-size:13px;line-height:1.5}.comparison-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px}.comparison{min-width:760px;width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed}.comparison th,.comparison td{padding:10px 12px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-align:left;vertical-align:top;overflow-wrap:anywhere}.comparison tr:last-child>*{border-bottom:0}.comparison tr>*:last-child{border-right:0}.comparison thead th{background:#f7f9f8}.comparison .field-name{width:150px;color:var(--muted);font-size:12px;font-weight:750}.comparison tr.diff>*{background:var(--diff)}.source-title{display:block;margin-bottom:4px;font-size:13px}.source-meta{display:block;min-height:28px;color:var(--muted);font-size:10px;font-weight:500;line-height:1.35}.empty-value{color:#a1adb3}button{border:0;border-radius:7px;font:inherit;font-weight:750;cursor:pointer}.small-button{width:100%;margin-top:8px;padding:7px 8px;background:#e7f0f3;color:var(--brand-dark);font-size:11px}.small-button.primary{background:var(--brand);color:#fff}.source-actions{display:grid;grid-template-columns:1fr 1fr;gap:6px}.editor-source{display:flex;align-items:center;gap:8px;margin:14px 0 16px;padding:10px 12px;border:1px solid #cddbe0;border-radius:8px;background:#f2f8fa;color:#34515e;font-size:13px}.editor-source strong{color:var(--ink)}fieldset{margin:0 0 20px;padding:0;border:0}legend{width:100%;margin-bottom:10px;padding-bottom:7px;border-bottom:1px solid var(--line);font-size:13px;font-weight:850;letter-spacing:.04em;text-transform:uppercase}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px 14px}.form-field.full{grid-column:1/-1}label{display:block;margin-bottom:5px;color:#42545e;font-size:12px;font-weight:700}input,textarea{width:100%;padding:9px 10px;border:1px solid #b9c6cc;border-radius:7px;background:#fff;color:var(--ink);font:inherit}textarea{min-height:82px;resize:vertical}.notes{min-height:90px}.action-bar{position:sticky;bottom:0;z-index:10;display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;margin:4px -18px -18px;padding:13px 18px;border-top:1px solid var(--line);border-radius:0 0 12px 12px;background:#fffffff2;backdrop-filter:blur(8px)}.action-group{display:flex;gap:8px}.button{padding:10px 14px}.button.primary{background:var(--brand);color:#fff}.button.secondary{border:1px solid #aebec5;background:#fff;color:var(--brand-dark)}.button.defer{background:#f3ebe8;color:var(--danger)}@media(max-width:980px){header{position:static}main{grid-template-columns:1fr;width:calc(100% - 20px)}.evidence-sticky{position:static}.notice-image{max-height:70vh}}@media(max-width:580px){.form-grid{grid-template-columns:1fr}.form-field.full{grid-column:auto}.gt-summary{display:block}.action-bar,.action-group{display:grid;grid-template-columns:1fr}.button{width:100%}}
</style></head><body>
<header><a href="/?view={{ return_view }}&amp;page={{ return_page }}{% if method_filter %}&amp;method={{ method_filter|urlencode }}{% endif %}">← Queue</a><div class="header-title"><strong>{{ detail.document.filename_stem }}</strong><span>{{ detail.document.source }} · {{ label_set }}</span></div>{% if detail.ground_truth %}<span class="state saved">GT available</span>{% else %}<span class="state pending">Needs review</span>{% endif %}</header>
<main><aside class="evidence"><div class="evidence-sticky"><section class="panel"><a href="/images/{{ detail.document.id }}" target="_blank"><img class="notice-image" src="/images/{{ detail.document.id }}" alt="Death notice {{ detail.document.filename_stem }}"></a></section><section class="panel panel-pad"><details><summary>OCR evidence</summary><div class="ocr-text">{{ detail.ocr.text if detail.ocr else 'No OCR text available.' }}</div></details></section></div></aside>
<div class="workspace"><section class="panel panel-pad"><div class="gt-summary"><div><h2>Ground truth status</h2>{% if detail.ground_truth %}<div class="gt-meta">Source: <strong>{{ initial_source_label }}</strong><br>Reviewer: {{ detail.ground_truth.reviewer or '—' }} · Updated {{ detail.ground_truth.updated_at }}</div>{% else %}<p class="help">No GT exists yet. Choose a method below or enter the values manually.</p>{% endif %}</div>{% if detail.ground_truth %}<span class="status-badge saved">Saved</span>{% else %}<span class="status-badge pending">Pending</span>{% endif %}</div></section>
<section class="panel panel-pad"><div class="section-heading"><div><h2>Compare methods</h2><p>Highlighted rows disagree. “Fill form” copies one complete source without saving it.</p></div></div>{% if comparison_sources %}<div class="comparison-wrap"><table class="comparison"><thead><tr><th class="field-name">Field</th>{% for source in comparison_sources %}<th><span class="source-title">{{ source.label }}</span>{% if source.kind == 'ground_truth' %}<span class="source-meta">Saved label · {{ detail.ground_truth.reviewer or 'unattributed' }}</span>{% else %}<span class="source-meta">{{ source.meta or source.source_kind }}<br>{{ source.created_at }}</span><div class="source-actions"><button class="small-button fill-form" type="button" data-candidate-id="{{ source.id }}">Fill form</button><form class="direct-approval" method="post" action="/candidates/{{ source.id }}/approve" data-source-label="{{ source.label }}" data-has-gt="{{ 'true' if detail.ground_truth else 'false' }}"><input type="hidden" name="method" value="{{ method_filter }}"><input type="hidden" name="continue_mode" value="next"><button class="small-button primary" type="submit">Approve as GT</button></form></div>{% endif %}</th>{% endfor %}</tr></thead><tbody>{% for column in columns %}<tr{% if column in different_fields %} class="diff"{% endif %}><th class="field-name">{{ field_labels[column] }}</th>{% for source in comparison_sources %}<td>{% if source.fields[column] %}{{ source.fields[column] }}{% else %}<span class="empty-value">—</span>{% endif %}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div>{% else %}<p class="help">No method results are available. You can still create GT manually.</p>{% endif %}</section>
<section class="panel panel-pad" id="editor"><div class="section-heading"><div><h2>Ground truth editor</h2><p>Saving replaces existing GT for this record; it never creates a duplicate.</p></div></div><form id="gt-form" method="post" action="/documents/{{ detail.document.id }}/labels"><input id="source-candidate-id" type="hidden" name="source_candidate_id" value="{{ source_candidate_id }}"><input type="hidden" name="method" value="{{ method_filter }}"><div class="editor-source">Started from: <strong id="editor-source-label">{{ initial_source_label }}</strong></div>
{% for group_title,group_fields in field_groups %}<fieldset><legend>{{ group_title }}</legend><div class="form-grid">{% for column in group_fields %}<div class="form-field{% if column in long_fields %} full{% endif %}"><label for="{{ column }}">{{ field_labels[column] }}</label>{% if column in long_fields %}<textarea id="{{ column }}" name="{{ column }}">{{ fields[column] }}</textarea>{% else %}<input id="{{ column }}" name="{{ column }}" value="{{ fields[column] }}">{% endif %}</div>{% endfor %}</div></fieldset>{% endfor %}<label for="review_notes">Review notes</label><textarea class="notes" id="review_notes" name="review_notes">{{ detail.ground_truth.review_notes if detail.ground_truth else '' }}</textarea><div class="action-bar"><div class="action-group"><button class="button primary" type="submit" name="continue_mode" value="next">Save &amp; next</button><button class="button secondary" type="submit" name="continue_mode" value="stay">Save</button></div>{% if not detail.ground_truth %}<button class="button defer" type="submit" form="needs-review-form">Needs review — next</button>{% endif %}</div></form>{% if not detail.ground_truth %}<form id="needs-review-form" method="post" action="/documents/{{ detail.document.id }}/needs-review"><input type="hidden" name="method" value="{{ method_filter }}"></form>{% endif %}</section></div></main>
<script id="candidate-data" type="application/json">{{ candidate_payload|tojson }}</script><script>
(()=>{const form=document.getElementById("gt-form");const data=JSON.parse(document.getElementById("candidate-data").textContent);const candidates=new Map(data.map(c=>[String(c.id),c]));const sourceId=document.getElementById("source-candidate-id");const sourceLabel=document.getElementById("editor-source-label");let dirty=false;form.addEventListener("input",()=>{dirty=true});form.addEventListener("submit",()=>{dirty=false});document.querySelectorAll(".fill-form").forEach(button=>button.addEventListener("click",()=>{const candidate=candidates.get(button.dataset.candidateId);if(!candidate)return;if(dirty&&!window.confirm(`Discard unsaved edits and fill the form from ${candidate.label}?`))return;Object.entries(candidate.fields).forEach(([name,value])=>{const field=form.elements.namedItem(name);if(field)field.value=value});sourceId.value=candidate.id;sourceLabel.textContent=candidate.label;dirty=true;document.getElementById("editor").scrollIntoView({block:"start"})}));document.querySelectorAll(".direct-approval").forEach(approval=>approval.addEventListener("submit",event=>{if(approval.dataset.hasGt==="true"&&!window.confirm(`Replace current GT with ${approval.dataset.sourceLabel}?`))event.preventDefault()}));window.addEventListener("beforeunload",event=>{if(!dirty)return;event.preventDefault();event.returnValue=""})})();
</script></body></html>"""
