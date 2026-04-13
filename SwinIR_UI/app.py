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
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
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
LQ_GENERATOR_ROOT = PROJECT_ROOT / "lq_generator"
LQ_CONFIG_PATH = LQ_GENERATOR_ROOT / "degradation_config.yml"
BATCH_TASKS_LOCK = threading.Lock()
BATCH_TASKS: dict[str, dict] = {}
MAX_BATCH_TASKS = 100

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
from model_registry import get_model_options, get_default_model_key  # noqa: E402


# 从 model_zoo 目录动态扫描可用模型，不再硬编码
MODEL_OPTIONS = get_model_options(PROJECT_ROOT / "model_zoo")
if not MODEL_OPTIONS:
    raise FileNotFoundError(
        f"model_zoo 目录 ({PROJECT_ROOT / 'model_zoo'}) 中没有找到可用的模型文件。"
    )
DEFAULT_MODEL_KEY = get_default_model_key(MODEL_OPTIONS)

WORKFLOW_OPTIONS = {
    "direct_lq_to_sr": {
        "label": "低质量图 -> 超分",
        "needs_lq_generator": False,
    },
    "real_hq_to_lq_to_sr": {
        "label": "真实图 -> 生成低质图 -> 超分",
        "needs_lq_generator": True,
    },
}
DEFAULT_WORKFLOW_KEY = "direct_lq_to_sr"


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
        large_model=cfg.get("large_model", False),
        model_path=str(cfg["model_path"]),
        folder_lq=None,
        folder_gt=None,
        tile=256,
        tile_overlap=32,
    )


def _resolve_config_path(base_dir: Path, raw_path: str | None) -> str | None:
    if not raw_path:
        return raw_path
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


class LQGeneratorService:
    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()
        self._init_error: Exception | None = None
        self._available = bool(LQ_CONFIG_PATH.exists())

    @property
    def is_available(self) -> bool:
        return self._available

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._init_error is not None:
            raise RuntimeError(str(self._init_error))
        if not self._available:
            raise FileNotFoundError(f"Cannot find lq_generator config: {LQ_CONFIG_PATH}")

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            if self._init_error is not None:
                raise RuntimeError(str(self._init_error))

            try:
                import yaml
                from lq_generator.degradation_pipeline import RealESRGANDegradation

                with LQ_CONFIG_PATH.open("r", encoding="utf-8") as f:
                    opt = yaml.safe_load(f)

                config_dir = LQ_CONFIG_PATH.parent
                if "io" in opt:
                    for key in ("input_dir", "output_dir"):
                        if key in opt["io"] and opt["io"][key]:
                            opt["io"][key] = _resolve_config_path(config_dir, opt["io"][key])
                if "log" in opt and "log_path" in opt["log"] and opt["log"]["log_path"]:
                    opt["log"]["log_path"] = _resolve_config_path(config_dir, opt["log"]["log_path"])

                device = opt.get("device", "cuda")
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                self._pipeline = RealESRGANDegradation(opt, device=device)
                return self._pipeline
            except Exception as exc:
                self._init_error = exc
                raise RuntimeError(
                    "Failed to initialize lq_generator. "
                    "Ensure dependencies (e.g. basicsr, pyyaml) are installed."
                ) from exc

    def degrade_bgr(self, image_bgr: np.ndarray, scale: int) -> np.ndarray:
        pipeline = self._load_pipeline()
        with self._lock:
            pipeline.scale = scale
            if image_bgr.ndim == 2:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            elif image_bgr.shape[2] == 4:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            lq_tensor = pipeline.degrade(tensor)

        lq_rgb = lq_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        lq_rgb = np.clip(lq_rgb, 0, 1)
        lq_rgb = (lq_rgb * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(lq_rgb, cv2.COLOR_RGB2BGR)


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

    def process_bgr(self, img_bgr: np.ndarray, model_key: str, tile: int | None) -> np.ndarray:
        return self._infer_tensor(img_bgr, model_key, tile)

    def process_single(self, image_bytes: bytes, model_key: str, tile: int | None) -> np.ndarray:
        data = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Invalid image file.")
        return self.process_bgr(img, model_key, tile)


def _parse_tile(raw_tile: str | None) -> int | None:
    if not raw_tile:
        return 256
    value = int(raw_tile)
    if value == 0:
        return None
    if value < 32 or value > 1024:
        raise ValueError("Tile must be 0 or between 32 and 1024.")
    return value


def _parse_workflow(raw_workflow: str | None) -> str:
    workflow_mode = (raw_workflow or DEFAULT_WORKFLOW_KEY).strip()
    if workflow_mode not in WORKFLOW_OPTIONS:
        raise ValueError("Unsupported workflow mode.")
    return workflow_mode


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _normalize_datetime_filter(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if len(value) == 16:
        value += ":00"
    parsed = dt.datetime.fromisoformat(value)
    return parsed.isoformat(timespec="seconds")


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


def _cleanup_batch_tasks() -> None:
    with BATCH_TASKS_LOCK:
        if len(BATCH_TASKS) <= MAX_BATCH_TASKS:
            return
        finished_ids = [
            run_id
            for run_id, task in BATCH_TASKS.items()
            if task["status"] in {"success", "error", "canceled"}
        ]
        for run_id in finished_ids[: max(0, len(BATCH_TASKS) - MAX_BATCH_TASKS)]:
            BATCH_TASKS.pop(run_id, None)


def _cleanup_old_results(max_age_days: int = 7) -> int:
    """清理超过 max_age_days 天的结果目录。"""
    import shutil
    cleaned = 0
    cutoff = dt.datetime.now() - dt.timedelta(days=max_age_days)
    for sub in ("single", "batch"):
        parent = RESULTS_ROOT / sub
        if not parent.exists():
            continue
        for entry in parent.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = dt.datetime.fromtimestamp(entry.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(entry)
                    cleaned += 1
            except OSError:
                continue
    return cleaned


def _cleanup_old_logs(max_days: int = 30) -> int:
    """清理超过 max_days 天的推理日志记录。"""
    cutoff = (dt.datetime.now() - dt.timedelta(days=max_days)).isoformat(timespec="seconds")
    try:
        with LOG_DB_LOCK:
            with sqlite3.connect(LOG_DB_PATH) as conn:
                cursor = conn.execute("DELETE FROM inference_logs WHERE created_at < ?", (cutoff,))
                return cursor.rowcount
    except sqlite3.Error:
        return 0


def _build_batch_task(run_id: str, model_key: str, model_label: str, tile: int | None, total_uploaded: int, output_dir: Path) -> dict:
    return {
        "run_id": run_id,
        "status": "queued",
        "model_key": model_key,
        "model_label": model_label,
        "tile": tile,
        "device": str(service.device),
        "total_uploaded": total_uploaded,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "preview_images": [],
        "output_dir": str(output_dir),
        "elapsed_ms": None,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "cancel_requested": False,
    }


def _snapshot_batch_task(run_id: str) -> dict | None:
    with BATCH_TASKS_LOCK:
        task = BATCH_TASKS.get(run_id)
        if task is None:
            return None
        return {
            "run_id": task["run_id"],
            "status": task["status"],
            "model_key": task["model_key"],
            "model_label": task["model_label"],
            "tile": task["tile"],
            "device": task["device"],
            "total_uploaded": task["total_uploaded"],
            "processed": task["processed"],
            "skipped": task["skipped"],
            "failed": task["failed"],
            "errors": task["errors"][:20],
            "preview_images": task["preview_images"][:8],
            "output_dir": task["output_dir"],
            "elapsed_ms": task["elapsed_ms"],
            "started_at": task["started_at"],
            "updated_at": task["updated_at"],
            "cancel_requested": task["cancel_requested"],
            "done": task["status"] in {"success", "error", "canceled"},
        }


def _run_batch_task(run_id: str, files_payload: list[dict], model_key: str, model_label: str, tile: int | None, batch_dir: Path) -> None:
    started = dt.datetime.now()
    with BATCH_TASKS_LOCK:
        task = BATCH_TASKS[run_id]
        task["status"] = "running"
        task["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")

    for payload in files_payload:
        with BATCH_TASKS_LOCK:
            task = BATCH_TASKS.get(run_id)
            if task is None:
                return
            if task["cancel_requested"]:
                task["status"] = "canceled"
                task["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
                break

        filename = payload.get("name", "")
        if not filename:
            with BATCH_TASKS_LOCK:
                BATCH_TASKS[run_id]["skipped"] += 1
                BATCH_TASKS[run_id]["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            continue

        relative_name = _normalize_upload_name(filename)
        if not _is_allowed_file(relative_name):
            with BATCH_TASKS_LOCK:
                BATCH_TASKS[run_id]["skipped"] += 1
                BATCH_TASKS[run_id]["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            continue

        try:
            out = service.process_single(payload["bytes"], model_key=model_key, tile=tile)
        except Exception as exc:
            with BATCH_TASKS_LOCK:
                BATCH_TASKS[run_id]["errors"].append(f"{relative_name}: {exc}")
                BATCH_TASKS[run_id]["failed"] += 1
                BATCH_TASKS[run_id]["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            continue

        rel_input = Path(relative_name)
        output_stem = rel_input.with_suffix("")
        output_path = batch_dir / output_stem.parent / f"{output_stem.name}_SwinIR.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not _save_image(output_path, out):
            with BATCH_TASKS_LOCK:
                BATCH_TASKS[run_id]["errors"].append(f"{relative_name}: save failed")
                BATCH_TASKS[run_id]["failed"] += 1
                BATCH_TASKS[run_id]["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            continue

        with BATCH_TASKS_LOCK:
            BATCH_TASKS[run_id]["processed"] += 1
            if len(BATCH_TASKS[run_id]["preview_images"]) < 8:
                BATCH_TASKS[run_id]["preview_images"].append(
                    {
                        "name": output_path.name,
                        "url": "/results/" + output_path.relative_to(RESULTS_ROOT).as_posix(),
                    }
                )
            BATCH_TASKS[run_id]["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with BATCH_TASKS_LOCK:
        task = BATCH_TASKS.get(run_id)
        if task is None:
            return

        elapsed_ms = int((dt.datetime.now() - started).total_seconds() * 1000)
        task["elapsed_ms"] = elapsed_ms
        task["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")

        if task["status"] != "canceled":
            task["status"] = "success" if task["processed"] > 0 else "error"

        error_message = "; ".join(task["errors"][:3]) if task["errors"] else None
        status = task["status"]
        device = task["device"]
        total_uploaded = task["total_uploaded"]
        processed = task["processed"]
        skipped = task["skipped"]
        output_dir = task["output_dir"]

    _insert_inference_log(
        run_id=run_id,
        mode="batch",
        status=status,
        model_key=model_key,
        model_label=model_label,
        tile=tile,
        device=device,
        total_uploaded=total_uploaded,
        processed=processed,
        skipped=skipped,
        elapsed_ms=elapsed_ms,
        output_dir=output_dir,
        error_message=error_message,
    )


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
    created_from: str = "",
    created_to: str = "",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
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
    if created_from:
        where_parts.append("created_at >= ?")
        params.append(created_from)
    if created_to:
        where_parts.append("created_at <= ?")
        params.append(created_to)

    where_sql = ""
    if where_parts:
        where_sql = " WHERE " + " AND ".join(where_parts)

    sql = "SELECT * FROM inference_logs" + where_sql + " ORDER BY id DESC LIMIT ? OFFSET ?"
    select_params = [*params, limit, offset]
    count_sql = "SELECT COUNT(*) FROM inference_logs" + where_sql

    with LOG_DB_LOCK:
        with sqlite3.connect(LOG_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, select_params).fetchall()
            total = int(conn.execute(count_sql, params).fetchone()[0])
    return [dict(row) for row in rows], total


for p in (RESULTS_ROOT, UPLOADS_ROOT, LOGS_ROOT):
    p.mkdir(parents=True, exist_ok=True)
_init_log_db()

service = SwinIRService()
lq_service = LQGeneratorService()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload


@app.errorhandler(413)
def too_large(e):
    return jsonify({"ok": False, "error": "上传文件过大，最大支持 50MB。"}), 413


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
            "workflows": [{"key": k, "label": v["label"]} for k, v in WORKFLOW_OPTIONS.items()],
            "default_workflow": DEFAULT_WORKFLOW_KEY,
            "lq_generator_ready": lq_service.is_available,
        }
    )


@app.get("/api/health")
def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    return jsonify({
        "ok": True,
        "status": "healthy",
        "device": str(service.device),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "lq_generator_ready": lq_service.is_available,
        "models_loaded": list(service._models.keys()),
    })


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
    except ValueError as exc:
        return _json_error(str(exc))
    try:
        workflow_mode = _parse_workflow(request.form.get("workflow_mode"))
    except ValueError as exc:
        return _json_error(str(exc))

    if WORKFLOW_OPTIONS[workflow_mode]["needs_lq_generator"] and not lq_service.is_available:
        return _json_error("lq_generator is not ready. Please check degradation_config.yml and dependencies.", status=503)

    run_id = f"{_timestamp()}_{uuid.uuid4().hex[:8]}"
    model_label = MODEL_OPTIONS[model_key]["label"]
    workflow_label = WORKFLOW_OPTIONS[workflow_mode]["label"]
    output_dir = RESULTS_ROOT / "single" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(Path(file.filename).stem) or "image"
    lq_output_path = output_dir / f"{safe_name}_LQ.png"
    output_path = output_dir / f"{safe_name}_SwinIR.png"

    image_bytes = file.read()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return _json_error("Invalid image file.", status=400)
    input_h, input_w = decoded.shape[:2]
    lq_image: np.ndarray | None = None

    try:
        started = dt.datetime.now()
        if workflow_mode == "real_hq_to_lq_to_sr":
            model_scale = MODEL_OPTIONS[model_key]["scale"]
            lq_image = lq_service.degrade_bgr(decoded, scale=model_scale)
            out = service.process_bgr(lq_image, model_key=model_key, tile=tile)
        else:
            out = service.process_bgr(decoded, model_key=model_key, tile=tile)
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
            error_message=f"[{workflow_mode}] {exc}",
        )
        return _json_error(str(exc), status=500)

    if lq_image is not None and not _save_image(lq_output_path, lq_image):
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
            error_message=f"[{workflow_mode}] Failed to save generated LQ image.",
        )
        return _json_error("Failed to save generated LQ image.", status=500)

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
            error_message=f"[{workflow_mode}] Failed to save output image.",
        )
        return _json_error("Failed to save output image.", status=500)

    lq_output_url = ""
    lq_saved_to = ""
    lq_w = input_w
    lq_h = input_h
    if lq_image is not None:
        lq_output_url = "/results/" + lq_output_path.relative_to(RESULTS_ROOT).as_posix()
        lq_saved_to = str(lq_output_path)
        lq_h, lq_w = lq_image.shape[:2]

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
            "workflow_mode": workflow_mode,
            "workflow_label": workflow_label,
            "output_url": output_url,
            "saved_to": str(output_path),
            "lq_output_url": lq_output_url,
            "lq_saved_to": lq_saved_to,
            "model": model_label,
            "device": str(service.device),
            "elapsed_ms": elapsed_ms,
            "run_id": run_id,
            "input_size": {"width": input_w, "height": input_h},
            "lq_size": {"width": int(lq_w), "height": int(lq_h)},
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
    except ValueError as exc:
        return _json_error(str(exc))

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


@app.post("/api/process-batch/start")
def process_batch_start():
    files = request.files.getlist("files")
    if not files:
        return _json_error("No files uploaded.")

    model_key = request.form.get("model_key", DEFAULT_MODEL_KEY)
    if model_key not in MODEL_OPTIONS:
        return _json_error("Unsupported model key.")
    model_label = MODEL_OPTIONS[model_key]["label"]

    try:
        tile = _parse_tile(request.form.get("tile"))
    except ValueError as exc:
        return _json_error(str(exc))

    run_id = f"{_timestamp()}_{uuid.uuid4().hex[:8]}"
    batch_dir = RESULTS_ROOT / "batch" / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    files_payload = [{"name": f.filename or "", "bytes": f.read()} for f in files]
    task = _build_batch_task(
        run_id=run_id,
        model_key=model_key,
        model_label=model_label,
        tile=tile,
        total_uploaded=len(files_payload),
        output_dir=batch_dir,
    )
    with BATCH_TASKS_LOCK:
        BATCH_TASKS[run_id] = task
    _cleanup_batch_tasks()

    worker = threading.Thread(
        target=_run_batch_task,
        args=(run_id, files_payload, model_key, model_label, tile, batch_dir),
        daemon=True,
    )
    worker.start()

    return jsonify(
        {
            "ok": True,
            "run_id": run_id,
            "status": "queued",
            "total_uploaded": len(files_payload),
            "output_dir": str(batch_dir),
            "model": model_label,
        }
    )


@app.get("/api/process-batch/status/<run_id>")
def process_batch_status(run_id: str):
    task = _snapshot_batch_task(run_id)
    if task is None:
        return _json_error("run_id not found.", status=404)
    return jsonify({"ok": True, **task})


@app.post("/api/process-batch/cancel/<run_id>")
def process_batch_cancel(run_id: str):
    with BATCH_TASKS_LOCK:
        task = BATCH_TASKS.get(run_id)
        if task is None:
            return _json_error("run_id not found.", status=404)
        if task["status"] in {"success", "error", "canceled"}:
            return jsonify({"ok": True, "run_id": run_id, "status": task["status"], "cancel_requested": False})
        task["cancel_requested"] = True
        task["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")

    return jsonify({"ok": True, "run_id": run_id, "status": "running", "cancel_requested": True})


@app.get("/api/logs")
def logs():
    run_id = request.args.get("run_id", "").strip()
    mode = request.args.get("mode", "").strip()
    status = request.args.get("status", "").strip()
    created_from_raw = request.args.get("created_from", "").strip()
    created_to_raw = request.args.get("created_to", "").strip()
    limit_raw = request.args.get("limit", "50").strip()
    offset_raw = request.args.get("offset", "0").strip()

    if mode and mode not in {"single", "batch"}:
        return _json_error("mode must be single or batch.")
    if status and status not in {"success", "error", "canceled"}:
        return _json_error("status must be success, error, or canceled.")

    try:
        created_from = _normalize_datetime_filter(created_from_raw) if created_from_raw else ""
        created_to = _normalize_datetime_filter(created_to_raw) if created_to_raw else ""
    except ValueError:
        return _json_error("created_from/created_to must be valid ISO datetime.")
    if created_from and created_to and created_from > created_to:
        return _json_error("created_from cannot be later than created_to.")

    try:
        limit = int(limit_raw)
    except ValueError:
        return _json_error("limit must be an integer.")
    if limit < 1 or limit > 200:
        return _json_error("limit must be between 1 and 200.")
    try:
        offset = int(offset_raw)
    except ValueError:
        return _json_error("offset must be an integer.")
    if offset < 0:
        return _json_error("offset must be >= 0.")

    try:
        records, total = _query_inference_logs(
            run_id=run_id,
            mode=mode,
            status=status,
            created_from=created_from,
            created_to=created_to,
            limit=limit,
            offset=offset,
        )
    except sqlite3.Error as exc:
        return _json_error(f"Failed to query logs: {exc}", status=500)
    return jsonify(
        {
            "ok": True,
            "count": len(records),
            "total": total,
            "limit": limit,
            "offset": offset,
            "records": records,
        }
    )


@app.post("/api/cleanup")
def cleanup():
    max_result_days = int(request.args.get("max_result_days", "7"))
    max_log_days = int(request.args.get("max_log_days", "30"))
    results_cleaned = _cleanup_old_results(max_age_days=max_result_days)
    logs_cleaned = _cleanup_old_logs(max_days=max_log_days)
    return jsonify({
        "ok": True,
        "results_dirs_cleaned": results_cleaned,
        "log_records_cleaned": logs_cleaned,
    })


@app.get("/results/<path:relative_path>")
def serve_results(relative_path: str):
    return send_from_directory(RESULTS_ROOT, relative_path)


if __name__ == "__main__":
    host = os.getenv("SWINIR_HOST", "127.0.0.1")
    port = int(os.getenv("SWINIR_PORT", "7860"))
    debug = os.getenv("SWINIR_DEBUG", "false").lower() in ("true", "1", "yes")
    app.run(host=host, port=port, debug=debug)
