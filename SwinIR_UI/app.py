from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys
import threading
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename


APP_ROOT = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    env_root = os.getenv("SWINIR_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    candidates = [
        APP_ROOT.parent / "SwinIR_model",      # New layout: GraduationProject/SwinIR_model + SwinIR_UI
        APP_ROOT.parent / "GraduationProject",  # Legacy layout: sibling SwinIR_UI + GraduationProject
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


PROJECT_ROOT = _resolve_project_root()
RESULTS_ROOT = APP_ROOT / "results"
UPLOADS_ROOT = APP_ROOT / "uploads"
LOGS_ROOT = APP_ROOT / "logs"
LOG_DB_PATH = LOGS_ROOT / "inference_logs.db"
LOG_DB_LOCK = threading.Lock()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Reuse existing SwinIR model definition and tile inference utility.
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"Cannot find SwinIR project root at: {PROJECT_ROOT}. "
        "Set SWINIR_PROJECT_ROOT to the correct absolute path if needed."
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from main_test_swinir import define_model, test  # noqa: E402


MODEL_OPTIONS = {
    "x2_psnr": {
        "label": "Real-SR x2 (PSNR)",
        "scale": 2,
        "model_path": PROJECT_ROOT
        / "model_zoo"
        / "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR-with-dict-keys-params-and-params_ema.pth",
    },
    "x2_gan": {
        "label": "Real-SR x2 (GAN)",
        "scale": 2,
        "model_path": PROJECT_ROOT
        / "model_zoo"
        / "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN-with-dict-keys-params-and-params_ema.pth",
    },
    "x4_psnr": {
        "label": "Real-SR x4 (PSNR)",
        "scale": 4,
        "model_path": PROJECT_ROOT
        / "model_zoo"
        / "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR-with-dict-keys-params-and-params_ema.pth",
    },
}
DEFAULT_MODEL_KEY = "x2_psnr"


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_upload_name(name: str) -> str:
    clean_name = name.replace("\\", "/").strip("/")
    parts = [secure_filename(p) for p in PurePosixPath(clean_name).parts if p not in ("", ".", "..")]
    parts = [p for p in parts if p]
    if not parts:
        return f"file_{uuid.uuid4().hex[:8]}.png"
    return "/".join(parts)


def _is_allowed_file(path_or_name: str) -> bool:
    return Path(path_or_name).suffix.lower() in ALLOWED_EXTENSIONS


def _build_args(model_key: str) -> SimpleNamespace:
    cfg = MODEL_OPTIONS[model_key]
    return SimpleNamespace(
        task="real_sr",
        scale=cfg["scale"],
        noise=15,
        jpeg=40,
        training_patch_size=64,
        large_model=False,
        model_path=str(cfg["model_path"]),
        folder_lq=None,
        folder_gt=None,
        tile=256,
        tile_overlap=32,
    )


class SwinIRService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _load_model(self, model_key: str) -> None:
        if model_key in self._models:
            return
        cfg = MODEL_OPTIONS[model_key]
        model_path = Path(cfg["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        args = _build_args(model_key)
        model = define_model(args).to(self.device)
        model.eval()
        self._models[model_key] = {"model": model, "args": args}

    def _infer_tensor(self, img_bgr: np.ndarray, model_key: str, tile: int | None) -> np.ndarray:
        with self._lock:
            self._load_model(model_key)
            bundle = self._models[model_key]
        model = bundle["model"]
        base_args = bundle["args"]

        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        img = img_bgr.astype(np.float32) / 255.0
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        args = SimpleNamespace(**vars(base_args))
        args.tile = tile if tile and tile > 0 else None
        window_size = 8

        with torch.no_grad():
            _, _, h_old, w_old = img.size()
            h_pad = (window_size - h_old % window_size) % window_size
            w_pad = (window_size - w_old % window_size) % window_size

            if h_pad > 0:
                img = torch.cat([img, torch.flip(img, [2])], dim=2)[:, :, : h_old + h_pad, :]
            if w_pad > 0:
                img = torch.cat([img, torch.flip(img, [3])], dim=3)[:, :, :, : w_old + w_pad]

            output = test(img, model, args, window_size)
            output = output[..., : h_old * args.scale, : w_old * args.scale]

        out = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
        out = (out * 255.0).round().astype(np.uint8)
        return out

    def process_single(self, image_bytes: bytes, model_key: str, tile: int | None) -> np.ndarray:
        data = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Invalid image file.")
        return self._infer_tensor(img, model_key, tile)


def _parse_tile(raw_tile: str | None) -> int | None:
    if not raw_tile:
        return 256
    value = int(raw_tile)
    if value <= 0:
        return None
    return value


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _save_image(path: Path, image: np.ndarray) -> bool:
    """Save image robustly on Windows paths (including non-ASCII paths)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(path), image):
        return True

    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False

    try:
        path.write_bytes(encoded.tobytes())
        return True
    except OSError:
        return False


def _init_log_db() -> None:
    with LOG_DB_LOCK:
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_key TEXT,
                    model_label TEXT,
                    tile INTEGER,
                    device TEXT,
                    input_name TEXT,
                    total_uploaded INTEGER,
                    processed INTEGER,
                    skipped INTEGER,
                    elapsed_ms INTEGER,
                    output_path TEXT,
                    output_dir TEXT,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_inference_logs_run_id ON inference_logs(run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at ON inference_logs(created_at DESC)"
            )


def _insert_inference_log(
    *,
    run_id: str,
    mode: str,
    status: str,
    model_key: str | None = None,
    model_label: str | None = None,
    tile: int | None = None,
    device: str | None = None,
    input_name: str | None = None,
    total_uploaded: int | None = None,
    processed: int | None = None,
    skipped: int | None = None,
    elapsed_ms: int | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    error_message: str | None = None,
) -> None:
    try:
        with LOG_DB_LOCK:
            with sqlite3.connect(LOG_DB_PATH) as conn:
                conn.execute(
                    """
                    INSERT INTO inference_logs (
                        created_at, run_id, mode, status, model_key, model_label, tile, device,
                        input_name, total_uploaded, processed, skipped, elapsed_ms, output_path,
                        output_dir, error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dt.datetime.now().isoformat(timespec="seconds"),
                        run_id,
                        mode,
                        status,
                        model_key,
                        model_label,
                        tile,
                        device,
                        input_name,
                        total_uploaded,
                        processed,
                        skipped,
                        elapsed_ms,
                        output_path,
                        output_dir,
                        error_message,
                    ),
                )
    except sqlite3.Error:
        # Logging persistence should not break inference requests.
        return


def _query_inference_logs(
    *,
    run_id: str = "",
    mode: str = "",
    status: str = "",
    limit: int = 50,
) -> list[dict]:
    where_parts: list[str] = []
    params: list[object] = []

    if run_id:
        where_parts.append("run_id LIKE ?")
        params.append(f"%{run_id}%")
    if mode:
        where_parts.append("mode = ?")
        params.append(mode)
    if status:
        where_parts.append("status = ?")
        params.append(status)

    sql = "SELECT * FROM inference_logs"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with LOG_DB_LOCK:
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


for p in (RESULTS_ROOT, UPLOADS_ROOT, LOGS_ROOT):
    p.mkdir(parents=True, exist_ok=True)
_init_log_db()

service = SwinIRService()

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    model_list = [{"key": k, "label": v["label"]} for k, v in MODEL_OPTIONS.items()]
    return render_template("index.html", models=model_list, default_model=DEFAULT_MODEL_KEY)


@app.get("/api/models")
def models():
    return jsonify(
        {
            "ok": True,
            "models": [{"key": k, "label": v["label"]} for k, v in MODEL_OPTIONS.items()],
            "default_model": DEFAULT_MODEL_KEY,
            "device": str(service.device),
        }
    )


@app.post("/api/process-single")
def process_single():
    file = request.files.get("image")
    if file is None or not file.filename:
        return _json_error("No image uploaded.")

    model_key = request.form.get("model_key", DEFAULT_MODEL_KEY)
    if model_key not in MODEL_OPTIONS:
        return _json_error("Unsupported model key.")

    if not _is_allowed_file(file.filename):
        return _json_error("Unsupported image format.")

    try:
        tile = _parse_tile(request.form.get("tile"))
    except ValueError:
        return _json_error("Tile must be an integer.")

    run_id = f"{_timestamp()}_{uuid.uuid4().hex[:8]}"
    model_label = MODEL_OPTIONS[model_key]["label"]
    output_dir = RESULTS_ROOT / "single" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(Path(file.filename).stem) or "image"
    output_path = output_dir / f"{safe_name}_SwinIR.png"

    image_bytes = file.read()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return _json_error("Invalid image file.", status=400)
    input_h, input_w = decoded.shape[:2]

    try:
        started = dt.datetime.now()
        out = service.process_single(image_bytes, model_key=model_key, tile=tile)
        elapsed_ms = int((dt.datetime.now() - started).total_seconds() * 1000)
    except Exception as exc:
        _insert_inference_log(
            run_id=run_id,
            mode="single",
            status="error",
            model_key=model_key,
            model_label=model_label,
            tile=tile,
            device=str(service.device),
            input_name=file.filename or "",
            error_message=str(exc),
        )
        return _json_error(str(exc), status=500)

    if not _save_image(output_path, out):
        _insert_inference_log(
            run_id=run_id,
            mode="single",
            status="error",
            model_key=model_key,
            model_label=model_label,
            tile=tile,
            device=str(service.device),
            input_name=file.filename or "",
            elapsed_ms=elapsed_ms,
            error_message="Failed to save output image.",
        )
        return _json_error("Failed to save output image.", status=500)

    output_url = "/results/" + output_path.relative_to(RESULTS_ROOT).as_posix()
    _insert_inference_log(
        run_id=run_id,
        mode="single",
        status="success",
        model_key=model_key,
        model_label=model_label,
        tile=tile,
        device=str(service.device),
        input_name=file.filename or "",
        elapsed_ms=elapsed_ms,
        output_path=str(output_path),
    )
    return jsonify(
        {
            "ok": True,
            "output_url": output_url,
            "saved_to": str(output_path),
            "model": model_label,
            "device": str(service.device),
            "elapsed_ms": elapsed_ms,
            "run_id": run_id,
            "input_size": {"width": input_w, "height": input_h},
            "output_size": {"width": int(out.shape[1]), "height": int(out.shape[0])},
        }
    )


@app.post("/api/process-batch")
def process_batch():
    files = request.files.getlist("files")
    if not files:
        return _json_error("No files uploaded.")

    model_key = request.form.get("model_key", DEFAULT_MODEL_KEY)
    if model_key not in MODEL_OPTIONS:
        return _json_error("Unsupported model key.")
    model_label = MODEL_OPTIONS[model_key]["label"]

    try:
        tile = _parse_tile(request.form.get("tile"))
    except ValueError:
        return _json_error("Tile must be an integer.")

    run_id = f"{_timestamp()}_{uuid.uuid4().hex[:8]}"
    batch_dir = RESULTS_ROOT / "batch" / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    started = dt.datetime.now()
    total_uploaded = len(files)
    processed = 0
    skipped = 0
    errors: list[str] = []
    previews: list[dict] = []

    for f in files:
        if not f.filename:
            skipped += 1
            continue
        relative_name = _normalize_upload_name(f.filename)
        if not _is_allowed_file(relative_name):
            skipped += 1
            continue

        try:
            out = service.process_single(f.read(), model_key=model_key, tile=tile)
        except Exception as exc:
            errors.append(f"{relative_name}: {exc}")
            continue

        rel_input = Path(relative_name)
        output_stem = rel_input.with_suffix("")
        output_path = batch_dir / output_stem.parent / f"{output_stem.name}_SwinIR.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not _save_image(output_path, out):
            errors.append(f"{relative_name}: save failed")
            continue

        processed += 1
        if len(previews) < 8:
            previews.append(
                {
                    "name": output_path.name,
                    "url": "/results/" + output_path.relative_to(RESULTS_ROOT).as_posix(),
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_ms = int((dt.datetime.now() - started).total_seconds() * 1000)
    batch_status = "success" if processed > 0 else "error"
    error_message = "; ".join(errors[:3]) if errors else None
    _insert_inference_log(
        run_id=run_id,
        mode="batch",
        status=batch_status,
        model_key=model_key,
        model_label=model_label,
        tile=tile,
        device=str(service.device),
        total_uploaded=total_uploaded,
        processed=processed,
        skipped=skipped,
        elapsed_ms=elapsed_ms,
        output_dir=str(batch_dir),
        error_message=error_message,
    )
    return jsonify(
        {
            "ok": True,
            "run_id": run_id,
            "total_uploaded": total_uploaded,
            "processed": processed,
            "skipped": skipped,
            "errors": errors[:20],
            "output_dir": str(batch_dir),
            "preview_images": previews,
            "model": model_label,
            "elapsed_ms": elapsed_ms,
        }
    )


@app.get("/api/logs")
def logs():
    run_id = request.args.get("run_id", "").strip()
    mode = request.args.get("mode", "").strip()
    status = request.args.get("status", "").strip()
    limit_raw = request.args.get("limit", "50").strip()

    if mode and mode not in {"single", "batch"}:
        return _json_error("mode must be single or batch.")
    if status and status not in {"success", "error"}:
        return _json_error("status must be success or error.")
    try:
        limit = int(limit_raw)
    except ValueError:
        return _json_error("limit must be an integer.")
    if limit < 1 or limit > 200:
        return _json_error("limit must be between 1 and 200.")

    try:
        records = _query_inference_logs(run_id=run_id, mode=mode, status=status, limit=limit)
    except sqlite3.Error as exc:
        return _json_error(f"Failed to query logs: {exc}", status=500)
    return jsonify({"ok": True, "count": len(records), "records": records})


@app.get("/results/<path:relative_path>")
def serve_results(relative_path: str):
    return send_from_directory(RESULTS_ROOT, relative_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
