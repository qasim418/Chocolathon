#!/usr/bin/env python3
"""FastAPI service for Chocolathon inference and visual processing output."""
import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

API_DIR = Path(__file__).resolve().parent
CONFIGURED_ROOT = Path(os.getenv("CHOCOLATHON_ROOT", API_DIR.parents[1])).resolve()
SOURCE_RUNTIME = CONFIGURED_ROOT / "Application" / "Runtime"

if (SOURCE_RUNTIME / "chocolathon_inference.py").is_file():
    PROJECT_ROOT, INFERENCE_DIR = CONFIGURED_ROOT, SOURCE_RUNTIME
elif (CONFIGURED_ROOT / "chocolathon_inference.py").is_file():
    PROJECT_ROOT = INFERENCE_DIR = CONFIGURED_ROOT
else:
    raise RuntimeError(f"Chocolathon runtime not found under {CONFIGURED_ROOT}.")

sys.path.insert(0, str(INFERENCE_DIR))
from chocolathon_inference import Chocolathon, LAYOUTS

logging.basicConfig(level=os.getenv("CHOCOLATHON_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("chocolathon.api")

DEVICE = os.getenv("CHOCOLATHON_DEVICE", "cpu")
THREADS = int(os.getenv("CHOCOLATHON_THREADS", "1"))
BATCH_SIZE = int(os.getenv("CHOCOLATHON_BATCH_SIZE", "8"))
MAX_UPLOAD_BYTES = int(os.getenv("CHOCOLATHON_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_IMAGE_PIXELS = int(os.getenv("CHOCOLATHON_MAX_IMAGE_PIXELS", "50000000"))
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png"}

if DEVICE not in {"cpu", "cuda"}: raise RuntimeError("CHOCOLATHON_DEVICE must be cpu or cuda")
if THREADS < 1 or BATCH_SIZE < 1: raise RuntimeError("CHOCOLATHON_THREADS and CHOCOLATHON_BATCH_SIZE must be positive")

app = FastAPI(
    title="Chocolathon Inference API",
    version="1.1.0",
    description="Offline tray localization, chocolate classification and processing visualization.",
)
app.state.engine = None
app.state.started_at = None
app.state.inference_lock = asyncio.Lock()


@app.on_event("startup")
def load_engine():
    started = time.perf_counter()
    LOGGER.info("Loading Chocolathon models from %s", PROJECT_ROOT)
    app.state.engine = Chocolathon(
        PROJECT_ROOT,
        device=DEVICE,
        threads=THREADS,
        batch_size=BATCH_SIZE,
        empty_mode="original",
        geometry="uniform",
        localizer="segmentation",
    )
    app.state.smoke = app.state.engine.smoke_test()
    app.state.started_at = time.time()
    LOGGER.info("Models loaded in %.3f seconds", time.perf_counter()-started)


@app.get("/")
def service_info():
    frontend = API_DIR / "index.html"
    if frontend.is_file(): return FileResponse(frontend)
    raise HTTPException(status_code=503, detail="Frontend file index.html is missing")


@app.get("/api")
def api_info():
    return {
        "service": "Chocolathon Inference API",
        "version": app.version,
        "status": "ready" if app.state.engine is not None else "loading",
        "documentation": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    if app.state.engine is None: raise HTTPException(status_code=503, detail="Models are not loaded")
    return {
        "status": "ready",
        "device": DEVICE,
        "threads": THREADS,
        "batch_size": BATCH_SIZE,
        "supported_capacities": list(LAYOUTS),
        "automatic_capacities": [6, 16, 30, 50],
        "manual_capacity_10": 10 in LAYOUTS,
        "primary_localizer": "segmentation",
        "fallback_localizer": "routed_corner_regression",
        "filled_only_default": True,
        "uptime_seconds": time.time()-app.state.started_at,
        "artifact_validation": app.state.smoke["artifact_validation"],
    }


def validate_image(payload):
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid JPEG or PNG image") from error
    if width < 32 or height < 32: raise HTTPException(status_code=422, detail="Image dimensions are too small")
    if width*height > MAX_IMAGE_PIXELS: raise HTTPException(status_code=413, detail="Image exceeds the configured pixel limit")
    return width, height


def display_font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def image_data_url(image, max_side=1200, quality=82):
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")


def processing_visuals(result, rgb, tray):
    original = Image.fromarray(rgb)
    localized = original.copy()
    draw = ImageDraw.Draw(localized)
    corners = [tuple(map(float, point)) for point in result["corners"]]
    width = max(4, round(min(localized.size)/180))

    draw.line(corners+[corners[0]], fill=(255, 196, 52), width=width, joint="curve")
    radius = width*2
    corner_font = display_font(max(14, width*3))

    for index, (x, y) in enumerate(corners, 1):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 78, 55), outline="white", width=max(1, width//2))
        draw.text((x+radius+2, y-radius), str(index), fill="white", font=corner_font, stroke_width=2, stroke_fill="black")

    caption = f'{result["capacity"]} slots · {result["geometry_mode"].replace("automatic_", "")} localization'
    caption_font = display_font(max(18, round(min(localized.size)/32)))
    caption_box = draw.textbbox((0, 0), caption, font=caption_font, stroke_width=1)
    draw.rounded_rectangle((12, 12, caption_box[2]+34, caption_box[3]+32), radius=10, fill=(35, 20, 14))
    draw.text((22, 20), caption, font=caption_font, fill="white")

    rectified = Image.fromarray(tray)
    rows, columns = LAYOUTS[result["capacity"]]
    grid = rectified.copy()
    grid_draw = ImageDraw.Draw(grid)
    cell_width, cell_height = grid.width/columns, grid.height/rows
    line_width = max(2, round(min(cell_width, cell_height)/60))

    for column in range(1, columns):
        x = round(column*cell_width)
        grid_draw.line((x, 0, x, grid.height), fill=(255, 196, 52), width=line_width)

    for row in range(1, rows):
        y = round(row*cell_height)
        grid_draw.line((0, y, grid.width, y), fill=(255, 196, 52), width=line_width)

    number_font = display_font(max(14, round(min(cell_width, cell_height)/9)))
    for row in range(rows):
        for column in range(columns):
            x, y = round(column*cell_width)+6, round(row*cell_height)+5
            grid_draw.text((x, y), f"{row+1},{column+1}", fill="white", font=number_font, stroke_width=2, stroke_fill="black")

    labeled = rectified.copy()
    label_draw = ImageDraw.Draw(labeled)
    label_font = display_font(max(12, round(min(cell_width, cell_height)/13)))

    for slot in result["slots"]:
        row, column = int(slot["row"])-1, int(slot["column"])-1
        x0, y0 = round(column*cell_width), round(row*cell_height)
        x1, y1 = round((column+1)*cell_width), round((row+1)*cell_height)
        name = str(slot["flavor_name"]).replace("_", " ")
        label_height = max(28, round(cell_height*.22))

        label_draw.rectangle((x0, y1-label_height, x1, y1), fill=(35, 20, 14))
        while label_draw.textlength(name, font=label_font) > x1-x0-12 and len(name) > 8:
            name = name[:-2].rstrip()+"…"
        label_draw.text((x0+6, y1-label_height+5), name, fill="white", font=label_font)
        label_draw.rectangle((x0, y0, x1-1, y1-1), outline=(255, 196, 52), width=line_width)

    return {
        "input": image_data_url(original),
        "localized": image_data_url(localized),
        "grid": image_data_url(grid),
        "labeled": image_data_url(labeled),
    }


def run_prediction(path, capacity, filled_only):
    engine = app.state.engine
    if engine is None: raise RuntimeError("Models are not loaded")
    result, rgb, tray, _ = engine.predict(path, filled_only=filled_only, capacity_override=capacity)
    result["visualizations"] = processing_visuals(result, rgb, tray)
    return result


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="JPEG or PNG box photograph"),
    capacity: Optional[int] = Form(None, description="Known capacity: 6, 10, 16, 30 or 50"),
    filled_only: bool = Form(True, description="Assign a chocolate flavor to every slot"),
):
    request_id = str(uuid.uuid4())
    content_type = image.content_type
    original_filename = Path(image.filename or "upload").name

    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")
    if capacity is not None and capacity not in LAYOUTS:
        raise HTTPException(status_code=422, detail="Capacity must be 6, 10, 16, 30 or 50")

    payload = await image.read(MAX_UPLOAD_BYTES+1)
    await image.close()

    if not payload: raise HTTPException(status_code=422, detail="Uploaded image is empty")
    if len(payload) > MAX_UPLOAD_BYTES: raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit")
    width, height = validate_image(payload)

    temporary_path = None
    started = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(prefix="chocolathon_", suffix=ALLOWED_TYPES[content_type], delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        del payload

        async with app.state.inference_lock:
            result = await asyncio.to_thread(run_prediction, temporary_path, capacity, filled_only)

        result["request_id"] = request_id
        result["input"] = {
            "filename": original_filename,
            "content_type": content_type,
            "width": width,
            "height": height,
            "capacity_override": capacity,
        }
        result["api_seconds"] = time.perf_counter()-started
        result.pop("image", None)
        LOGGER.info("request=%s capacity=%s occupied=%s seconds=%.3f", request_id, result["capacity"], result["occupied"], result["api_seconds"])
        return result

    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as error:
        LOGGER.warning("request=%s rejected: %s", request_id, error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("request=%s inference failure", request_id)
        raise HTTPException(status_code=500, detail="Inference failed") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chocolathon_api:app", host=os.getenv("CHOCOLATHON_HOST", "127.0.0.1"), port=int(os.getenv("CHOCOLATHON_PORT", "8000")), reload=False)