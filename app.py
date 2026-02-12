import os
import sys
import math
import glob
import csv
import urllib.parse
import logging
import gzip
import json
import hashlib
import time
import secrets
import threading
from bisect import bisect_left, bisect_right
from typing import Dict, Tuple, Optional, List

from flask import (
    Flask, render_template, jsonify, request,
    make_response, send_file
)

# ─────────────────────────────────────────────────────────────
# Optional deps (tqdm / pyshp / geopandas / pillow)
# ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import shapefile
    PYSHAPEFILE_AVAILABLE = True
except Exception:
    PYSHAPEFILE_AVAILABLE = False
    shapefile = None

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False
    gpd = None

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except Exception:
    PILLOW_AVAILABLE = False
    Image = None


# ─────────────────────────────────────────────────────────────
# Basic env / encoding (keep simple)
# ─────────────────────────────────────────────────────────────
if sys.platform.startswith("win"):
    try:
        os.environ["PYTHONIOENCODING"] = "utf-8"
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Logging
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.INFO)
werkzeug_logger.addHandler(handler)


# ─────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────
def normalize_path_for_storage(path: str) -> str:
    if not path:
        return ""
    return os.path.normpath(path).replace("\\", "/")


def get_absolute_path(relative_or_absolute_path: str) -> str:
    """상대 경로면 project_root 기준 절대경로로 변환. 절대경로면 그대로(정규화)."""
    if not relative_or_absolute_path:
        return ""
    normalized = normalize_path_for_storage(relative_or_absolute_path)
    # os.path.isabs는 OS sep 기준이라 보정
    is_abs = os.path.isabs(normalized.replace("/", os.sep)) or normalized.startswith("/")
    if is_abs:
        # project_root prefix 중복 방지
        pr = normalize_path_for_storage(project_root)
        if normalized.startswith(pr):
            rel_part = normalized[len(pr):].lstrip("/")
            if rel_part:
                return normalize_path_for_storage(os.path.join(project_root, rel_part))
            return pr
        return normalized
    return normalize_path_for_storage(os.path.join(project_root, normalized))


# ─────────────────────────────────────────────────────────────
# Default folders (relative)
# ─────────────────────────────────────────────────────────────
UPLOAD_IMG_FOLDER = "static/img"
TRACK_FOLDER = "static/track"
POINT_FOLDER = "static/waypoint"
MATCH_FOLDER = "static/match"
TILE_FOLDER = "static/tileimg"
CACHE_FOLDER = "static/cache"  # ✅ points gzip cache / indexes

os.makedirs(get_absolute_path(CACHE_FOLDER), exist_ok=True)

folder_settings = {
    "img_folder": UPLOAD_IMG_FOLDER,
    "track_folder": TRACK_FOLDER,
    "point_folder": POINT_FOLDER,
    "match_folder": MATCH_FOLDER,
    "tile_folder": TILE_FOLDER,
}

# ─────────────────────────────────────────────────────────────
# Global caches (shared, safe for all clients)
# ─────────────────────────────────────────────────────────────
dataset_cache: Dict[str, List[dict]] = {}
dataset_cache_signature: Dict[str, Tuple[float, float, float, bool]] = {}

# ✅ Image index cache: dataset -> dict
image_index_cache: Dict[str, dict] = {}

# ✅ Dataset list cache
dataset_list_cache: Dict[str, object] = {
    "sig": None,
    "datasets": [],
}

# ✅ Tile root map cache: built once -> (dataset,img_name) -> base_dir
tile_root_map: Dict[Tuple[str, str], str] = {}
tile_root_map_sig: Optional[float] = None

# ─────────────────────────────────────────────────────────────
# ✅ Global active state (shared for all clients)
# ─────────────────────────────────────────────────────────────
ACTIVE_STATE_LOCK = threading.Lock()
ACTIVE_STATE = {
    "points": [],                # merged points
    "points_by_lat": [],         # (lat, lon, point) sorted by lat
    "point_lats": [],            # sorted lat array (for bisect)
    "current_dataset": None,     # "A,B" 형태
    "use_tiles": True,
    "sig_hash": None,
    "ts": 0.0,
}

DEFAULT_DATASETS = os.getenv("DEFAULT_DATASETS", "").strip()  # 예: "D20250821"
DEFAULT_USE_TILES = True


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0
    except Exception:
        return 0.0


def _set_active(points: List[dict], dataset_names: List[str], use_tiles: bool):
    points = points or []
    points_by_lat = []
    point_lats = []

    for p in points:
        lat = p.get("lat")
        lon = p.get("lon")
        if lat is None or lon is None:
            coords = p.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lat, lon = coords[0], coords[1]

        if lat is None or lon is None:
            continue

        try:
            points_by_lat.append((float(lat), float(lon), p))
        except Exception:
            continue

    points_by_lat.sort(key=lambda row: row[0])
    point_lats = [row[0] for row in points_by_lat]

    with ACTIVE_STATE_LOCK:
        ACTIVE_STATE["points"] = points
        ACTIVE_STATE["points_by_lat"] = points_by_lat
        ACTIVE_STATE["point_lats"] = point_lats
        ACTIVE_STATE["current_dataset"] = ",".join([str(x) for x in dataset_names]) if dataset_names else None
        ACTIVE_STATE["use_tiles"] = bool(use_tiles)
        ACTIVE_STATE["sig_hash"] = _datasets_sig_hash(dataset_names, use_tiles) if dataset_names else None
        ACTIVE_STATE["ts"] = time.time()


def _get_active_points_snapshot() -> List[dict]:
    with ACTIVE_STATE_LOCK:
        pts = ACTIVE_STATE.get("points") or []
        # 얕은 참조 반환(대부분 read-only 사용 가정). 불안하면 list(pts)로 복제.
        return pts


def _get_active_dataset_name() -> Optional[str]:
    with ACTIVE_STATE_LOCK:
        return ACTIVE_STATE.get("current_dataset")


def _get_active_points_by_lat_snapshot() -> Tuple[List[Tuple[float, float, dict]], List[float]]:
    with ACTIVE_STATE_LOCK:
        rows = ACTIVE_STATE.get("points_by_lat")
        lats = ACTIVE_STATE.get("point_lats")
        safe_rows = rows if isinstance(rows, list) else []
        safe_lats = lats if isinstance(lats, list) else []
        return safe_rows, safe_lats


def _query_active_points_in_bbox(bbox: str, limit: int = 5000) -> List[dict]:
    if not bbox:
        return []

    try:
        west, south, east, north = map(float, str(bbox).split(","))
    except Exception:
        return []

    points_by_lat, point_lats = _get_active_points_by_lat_snapshot()
    if not points_by_lat or not point_lats:
        return []

    safe_limit = max(1, int(limit))
    start_idx = bisect_left(point_lats, south)
    end_idx = bisect_right(point_lats, north)

    out = []
    for _lat, lon, point in points_by_lat[start_idx:end_idx]:
        if west <= lon <= east:
            out.append(point)
            if len(out) >= safe_limit:
                break

    return out


# ─────────────────────────────────────────────────────────────
# Util
# ─────────────────────────────────────────────────────────────
def get_node_index_file() -> str:
    return os.path.join(get_absolute_path(CACHE_FOLDER), "node_index.json.gz")


def get_bearing(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)
    y = math.sin(d_lon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lon)
    brng = math.atan2(y, x)
    deg = (math.degrees(brng) + 360) % 360
    return deg


def get_location_info(lat, lon):
    return None


def parse_date_from_gpx_time(gpx_time):
    try:
        if not gpx_time or not isinstance(gpx_time, str):
            return None
        date_part = gpx_time.split(" ")[0]
        if "/" in date_part:
            year, month, day = date_part.split("/")
        elif "-" in date_part:
            year, month, day = date_part.split("-")
        else:
            return None
        return {
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "year_month": f"{year}년 {int(month)}월",
            "full_date": date_part,
        }
    except Exception:
        return None


def get_image_folder_path(dataset_name: str) -> str:
    img_base = get_absolute_path(UPLOAD_IMG_FOLDER).rstrip("/\\")
    base_name = os.path.basename(img_base)
    if base_name == dataset_name:
        return img_base
    return os.path.join(img_base, dataset_name)


# ─────────────────────────────────────────────────────────────
# ✅ Tile root map (build once)
# ─────────────────────────────────────────────────────────────
def _build_tile_root_map(force: bool = False):
    """
    (dataset,img)->base_dir 를 한번만 구축.
    구조:
      TILE_FOLDER/dataset/img_name/level/y/x.jpg
    또는:
      TILE_FOLDER/img_name/level/y/x.jpg
    """
    global tile_root_map, tile_root_map_sig

    tile_base = get_absolute_path(TILE_FOLDER)
    sig = _safe_mtime(tile_base)

    if not force and tile_root_map_sig is not None and tile_root_map_sig == sig and tile_root_map:
        return

    tile_root_map = {}
    tile_root_map_sig = sig

    if not os.path.exists(tile_base):
        return

    # 1) TILE_FOLDER/dataset/img_name/...
    try:
        for ds in os.listdir(tile_base):
            ds_path = os.path.join(tile_base, ds)
            if not os.path.isdir(ds_path):
                continue
            try:
                for img_name in os.listdir(ds_path):
                    img_path = os.path.join(ds_path, img_name)
                    if os.path.isdir(img_path):
                        tile_root_map[(ds, img_name)] = img_path
            except Exception:
                pass
    except Exception:
        pass

    # 2) TILE_FOLDER/img_name/...
    try:
        for img_name in os.listdir(tile_base):
            img_path = os.path.join(tile_base, img_name)
            if os.path.isdir(img_path):
                tile_root_map[("*", img_name)] = img_path
    except Exception:
        pass


def _resolve_tile_base_dir(dataset_name: str, img_name: str) -> Optional[str]:
    _build_tile_root_map(force=False)
    if (dataset_name, img_name) in tile_root_map:
        return tile_root_map[(dataset_name, img_name)]
    if ("*", img_name) in tile_root_map:
        return tile_root_map[("*", img_name)]
    return None


def _has_tiles(dataset_name: str, img_name: str) -> bool:
    base_dir = _resolve_tile_base_dir(dataset_name, img_name)
    if not base_dir:
        return False
    level0 = os.path.join(base_dir, "0")
    try:
        if os.path.isdir(level0):
            for item in os.listdir(level0):
                if os.path.isdir(os.path.join(level0, item)):
                    return True
    except Exception:
        return False
    return False


# ─────────────────────────────────────────────────────────────
# ✅ Image index cache
# ─────────────────────────────────────────────────────────────
def _build_image_index(dataset_name: str) -> dict:
    """
    dataset 이미지 폴더 전체 walk는 비쌈.
    → dataset별 1회 구축 후 캐시에 저장
    """
    img_folder = get_image_folder_path(dataset_name)
    img_folder_abs = img_folder if os.path.exists(img_folder) else get_absolute_path(UPLOAD_IMG_FOLDER)

    image_extensions = (".png", ".jpg", ".jpeg")
    all_images: List[str] = []
    all_images_set = set()

    if os.path.exists(img_folder_abs):
        for root, _dirs, files in os.walk(img_folder_abs):
            for f in files:
                if not f.lower().endswith(image_extensions):
                    continue
                rel_path = os.path.relpath(os.path.join(root, f), img_folder_abs)
                norm = rel_path.replace("\\", "/")
                if norm not in all_images_set:
                    all_images.append(norm)
                    all_images_set.add(norm)

    basename_to_image: Dict[str, List[str]] = {}
    for img in all_images:
        bn = os.path.basename(img).lower()
        basename_to_image.setdefault(bn, []).append(img)

    static_img_path = os.path.join(project_root, "static", "img")
    static_img_path_norm = static_img_path.replace("\\", "/")
    img_folder_norm = img_folder_abs.replace("\\", "/")

    is_static_parent = False
    use_dataset_in_path = False
    try:
        if img_folder_norm.startswith(static_img_path_norm):
            is_static_parent = True
            rel_path = os.path.relpath(img_folder_abs, static_img_path).replace("\\", "/")
            if rel_path == dataset_name or rel_path.startswith(dataset_name + "/"):
                use_dataset_in_path = True
    except Exception:
        pass

    return {
        "sig": _safe_mtime(img_folder_abs),
        "img_folder_abs": img_folder_abs,
        "all_images": all_images,
        "all_images_set": all_images_set,
        "basename_to_image": basename_to_image,
        "is_static_parent": is_static_parent,
        "use_dataset_in_path": use_dataset_in_path,
    }


def get_image_index(dataset_name: str, force: bool = False) -> dict:
    if not force and dataset_name in image_index_cache:
        return image_index_cache[dataset_name]
    idx = _build_image_index(dataset_name)
    image_index_cache[dataset_name] = idx
    return idx


# ─────────────────────────────────────────────────────────────
# ✅ gzip json cache helpers for points response
# ─────────────────────────────────────────────────────────────
def _dataset_signature(dataset_name: str, use_tiles: bool) -> Tuple[float, float, float, bool]:
    shp_path = os.path.join(get_absolute_path(POINT_FOLDER), f"{dataset_name}.shp")
    csv_path = os.path.join(get_absolute_path(MATCH_FOLDER), f"{dataset_name}.csv")

    img_base = get_absolute_path(UPLOAD_IMG_FOLDER).rstrip("/\\")
    base_name = os.path.basename(img_base)
    if base_name == dataset_name:
        img_folder = img_base
    else:
        img_folder = os.path.join(img_base, dataset_name)

    return (_safe_mtime(img_folder), _safe_mtime(csv_path), _safe_mtime(shp_path), bool(use_tiles))


def _datasets_sig_hash(dataset_names: List[str], use_tiles: bool) -> str:
    names = sorted([str(x) for x in dataset_names])
    sigs = []
    for name in names:
        try:
            sigs.append((name, _dataset_signature(name, use_tiles)))
        except Exception:
            sigs.append((name, (0.0, 0.0, 0.0, bool(use_tiles))))
    raw = json.dumps(sigs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _points_cache_path(dataset_names: List[str], use_tiles: bool) -> str:
    key = "_".join(sorted([str(x) for x in dataset_names]))
    sig_hash = _datasets_sig_hash(dataset_names, use_tiles)
    filename = f"points_{key}_{sig_hash}_tiles{int(bool(use_tiles))}.json.gz"
    return os.path.join(get_absolute_path(CACHE_FOLDER), filename)


def _read_gzip_json(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_gzip_json(path: str, payload: dict):
    try:
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        app.logger.warning(f"[CACHE] gzip write failed: {e}")


def _gzip_response_json(payload: dict, status: int = 200):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    gz = gzip.compress(raw, compresslevel=6)
    resp = make_response(gz, status)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    resp.headers["Content-Encoding"] = "gzip"
    resp.headers["Vary"] = "Accept-Encoding"
    return resp


# ─────────────────────────────────────────────────────────────
# Node index (node_id -> dataset)
# ─────────────────────────────────────────────────────────────
def _node_index_sig() -> str:
    try:
        point_abs = get_absolute_path(POINT_FOLDER)
        shp_files = glob.glob(os.path.join(point_abs, "*.shp"))
        if not shp_files:
            return "empty"
        meta = []
        for fp in shp_files:
            meta.append((os.path.basename(fp), _safe_mtime(fp)))
        meta.sort(key=lambda x: x[0])
        raw = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]
    except Exception:
        return "unknown"


def _read_node_index() -> Optional[dict]:
    try:
        path = get_node_index_file()
        if not os.path.exists(path):
            return None
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "sig" not in data or "map" not in data:
            return None
        if not isinstance(data["map"], dict):
            return None
        return data
    except Exception:
        return None


def _write_node_index(sig: str, mapping: dict):
    try:
        path = get_node_index_file()
        tmp = path + ".tmp"
        payload = {"sig": sig, "map": mapping}
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        app.logger.warning(f"[NODE_INDEX] write failed: {e}")


def _extract_node_id_from_record(record, field_names: List[str]) -> Optional[str]:
    attrs = {}
    try:
        for i in range(min(len(field_names), len(record))):
            attrs[str(field_names[i]).lower()] = record[i]
    except Exception:
        attrs = {}

    for k in ["track_seg_point_id", "track_seg_point", "track_seg", "track_point_id", "track_id"]:
        v = attrs.get(k)
        if v:
            try:
                return str(v).strip()
            except Exception:
                pass

    image_file = None
    for k in ["image_file", "image_filename", "img_file", "img_name"]:
        v = attrs.get(k)
        if v:
            image_file = v
            break

    if image_file:
        try:
            base = os.path.basename(str(image_file))
            name_no_ext, _ = os.path.splitext(base)
            if "_" in name_no_ext:
                return name_no_ext.rsplit("_", 1)[-1].strip()
            return name_no_ext.strip()
        except Exception:
            pass

    try:
        if len(record) > 0 and record[0]:
            return str(record[0]).strip()
    except Exception:
        pass

    return None


def build_node_index(force: bool = False) -> dict:
    sig = _node_index_sig()
    existing = _read_node_index()
    if not force and existing and existing.get("sig") == sig:
        return existing

    mapping = {}
    point_abs = get_absolute_path(POINT_FOLDER)
    shp_files = glob.glob(os.path.join(point_abs, "*.shp"))
    shp_files.sort()

    if not shp_files:
        _write_node_index(sig, {})
        return {"sig": sig, "map": {}}

    if not PYSHAPEFILE_AVAILABLE:
        app.logger.warning("[NODE_INDEX] pyshp not available; node index cannot be built.")
        _write_node_index(sig, {})
        return {"sig": sig, "map": {}}

    start = time.time()
    for shp_path in shp_files:
        dataset_name = os.path.splitext(os.path.basename(shp_path))[0]
        try:
            sf = shapefile.Reader(shp_path)
            field_names = [f[0] for f in sf.fields[1:]]
            for sr in sf.shapeRecords():
                rec = sr.record
                node_id = _extract_node_id_from_record(rec, field_names)
                if not node_id:
                    continue
                node_id = str(node_id).strip()
                if node_id and node_id not in mapping:
                    mapping[node_id] = dataset_name
            try:
                sf.close()
            except Exception:
                pass
        except Exception as e:
            app.logger.warning(f"[NODE_INDEX] skip {dataset_name}: {e}")

    _write_node_index(sig, mapping)
    app.logger.info(f"[NODE_INDEX] built: {len(mapping)} ids, sig={sig}, elapsed={time.time() - start:.2f}s")
    return {"sig": sig, "map": mapping}


def resolve_dataset_by_node_id(node_id: str) -> Optional[str]:
    if not node_id:
        return None
    node_id = str(node_id).strip()

    sig = _node_index_sig()
    data = _read_node_index()
    if (not data) or (data.get("sig") != sig) or ("map" not in data):
        data = build_node_index(force=True)

    mapping = data.get("map", {})
    return mapping.get(node_id)


# ─────────────────────────────────────────────────────────────
# Shapefile loaders
# ─────────────────────────────────────────────────────────────
def load_points_with_pyshp(shp_path, dataset_name=None):
    if not PYSHAPEFILE_AVAILABLE:
        return []

    try:
        sf = shapefile.Reader(shp_path)
        field_names = [f[0] for f in sf.fields[1:]]
        pts = []

        for shape in (sf.shapeRecords() if not TQDM_AVAILABLE else tqdm(sf.shapeRecords(), desc="Reading points", unit="records")):
            geom = shape.shape
            record = shape.record

            attributes = {}
            attributes_lower = {}
            try:
                attributes = {field_names[i]: record[i] for i in range(min(len(field_names), len(record)))}
                attributes_lower = {str(k).lower(): v for k, v in attributes.items()}
            except Exception:
                pass

            if geom.shapeType == shapefile.POINT:
                lon, lat = geom.points[0]

                image_file = None
                track_seg_point_id = None

                for key in ["image_file", "image_filename", "img_file", "img_name"]:
                    if key in attributes and attributes[key]:
                        image_file = attributes[key]
                        break
                    if key in attributes_lower and attributes_lower[key]:
                        image_file = attributes_lower[key]
                        break

                for key in ["track_seg_point_id", "track_seg_point", "track_seg", "track_point_id", "track_id"]:
                    if key in attributes and attributes[key]:
                        track_seg_point_id = str(attributes[key]).strip()
                        break
                    if key in attributes_lower and attributes_lower[key]:
                        track_seg_point_id = str(attributes_lower[key]).strip()
                        break

                if image_file is None:
                    image_file = record[1] if len(record) > 1 else None
                if track_seg_point_id is None:
                    track_seg_point_id = str(record[0]).strip() if len(record) > 0 else None

                gpx_time = attributes.get("gpx_time") if attributes.get("gpx_time") else (record[2] if len(record) > 2 else None)
                elevation = record[6] if len(record) > 6 else 0
                time_value = record[2] if len(record) > 2 else None

                if not track_seg_point_id:
                    if image_file:
                        base = os.path.basename(str(image_file))
                        name_no_ext, _ = os.path.splitext(base)
                        track_seg_point_id = name_no_ext.rsplit("_", 1)[-1].strip() if "_" in name_no_ext else name_no_ext.strip()
                    else:
                        track_seg_point_id = f"node_{len(pts)}"

                pts.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "elevation": elevation,
                        "time": time_value,
                        "gpx_time": gpx_time,
                        "image_file": image_file,
                        "track_seg_point_id": track_seg_point_id,
                        "angle": 0,
                        "angle_add": 0,
                        "location_info": None,
                        "date_info": parse_date_from_gpx_time(gpx_time),
                    }
                )

        try:
            sf.close()
        except Exception:
            pass

        return pts
    except Exception as e:
        app.logger.error(f"pyshp point read failed: {e}")
        return []


def load_shapefile_with_geopandas(shp_path):
    if not GEOPANDAS_AVAILABLE:
        return []
    try:
        if not os.path.exists(shp_path):
            return []
        gdf = gpd.read_file(os.path.normpath(shp_path))
        lines = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type in ["LineString", "MultiLineString"]:
                coords = []
                if geom.geom_type == "LineString":
                    coords = [[float(c[1]), float(c[0])] for c in geom.coords]
                else:
                    for line in geom.geoms:
                        coords.extend([[float(c[1]), float(c[0])] for c in line.coords])
                if coords:
                    lines.append({"coordinates": coords, "properties": {"name": f"line_{len(lines)}", "id": len(lines)}})
        return lines
    except Exception:
        return []


def load_shapefile_with_pyshp(shp_path):
    if not PYSHAPEFILE_AVAILABLE:
        return load_shapefile_with_geopandas(shp_path) if GEOPANDAS_AVAILABLE else []

    try:
        sf = shapefile.Reader(os.path.normpath(shp_path))
        lines = []
        for shape in sf.shapeRecords():
            geom = shape.shape
            geom_type = geom.shapeType
            is_polyline = (
                geom_type in (3, 13, 23)
                or (hasattr(shapefile, "POLYLINE") and geom_type == shapefile.POLYLINE)
                or (hasattr(shapefile, "POLYLINEZ") and geom_type == shapefile.POLYLINEZ)
                or (hasattr(shapefile, "POLYLINEM") and geom_type == shapefile.POLYLINEM)
            )
            if not is_polyline:
                continue
            pts = geom.points if hasattr(geom, "points") else []
            parts = geom.parts if hasattr(geom, "parts") else []
            coords = []
            if pts:
                if parts:
                    parts_list = list(parts) + [len(pts)]
                    for j in range(len(parts_list) - 1):
                        a, b = parts_list[j], parts_list[j + 1]
                        coords.extend([[float(p[1]), float(p[0])] for p in pts[a:b]])
                else:
                    coords = [[float(p[1]), float(p[0])] for p in pts]
            if coords:
                lines.append({"coordinates": coords, "properties": {"name": f"line_{len(lines)}", "id": len(lines)}})
        try:
            sf.close()
        except Exception:
            pass
        return lines
    except Exception:
        return load_shapefile_with_geopandas(shp_path) if GEOPANDAS_AVAILABLE else []


# ─────────────────────────────────────────────────────────────
# ✅ FAST datasets scan (no full image walk per dataset)
# ─────────────────────────────────────────────────────────────
def _quick_has_any_image(folder: str, max_checks: int = 2000) -> Tuple[bool, int]:
    if not folder or not os.path.exists(folder):
        return False, 0

    exts = (".png", ".jpg", ".jpeg")
    count = 0
    checked = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            checked += 1
            if f.lower().endswith(exts):
                count += 1
            if checked >= max_checks:
                return (count > 0), count
    return (count > 0), count


def get_shapefile_datasets():
    datasets = []
    try:
        point_abs = get_absolute_path(POINT_FOLDER)
        point_files = glob.glob(os.path.join(point_abs, "*.shp"))

        sig_meta = []
        for point_file in point_files:
            base_name = os.path.splitext(os.path.basename(point_file))[0]
            track_file = os.path.join(get_absolute_path(TRACK_FOLDER), f"{base_name}.shp")
            img_folder = get_image_folder_path(base_name)
            if not (os.path.exists(img_folder) and os.path.isdir(img_folder)):
                img_folder = get_absolute_path(UPLOAD_IMG_FOLDER)

            sig_meta.append((
                base_name,
                _safe_mtime(point_file),
                _safe_mtime(track_file),
                _safe_mtime(img_folder),
            ))

        sig_meta.sort(key=lambda row: row[0])
        sig = hashlib.sha1(
            json.dumps(sig_meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]

        if dataset_list_cache.get("sig") == sig:
            return dataset_list_cache.get("datasets", [])

        for point_file in point_files:
            base_name = os.path.splitext(os.path.basename(point_file))[0]

            img_folder = get_image_folder_path(base_name)
            if not (os.path.exists(img_folder) and os.path.isdir(img_folder)):
                img_folder = get_absolute_path(UPLOAD_IMG_FOLDER)

            has_images, found_count = _quick_has_any_image(img_folder, max_checks=1500)

            track_file = os.path.join(get_absolute_path(TRACK_FOLDER), f"{base_name}.shp")
            has_track = os.path.exists(track_file)

            info = {
                "name": base_name,
                "point_file": os.path.basename(point_file),
                "track_file": os.path.basename(track_file) if has_track else None,
                "has_images": has_images,
                "has_track": has_track,
                "point_count": found_count,
                "point_count_estimated": True,
                "is_complete": bool(has_images),
                "data_type": "shapefile",
            }
            if info["is_complete"]:
                datasets.append(info)

        dataset_list_cache["sig"] = sig
        dataset_list_cache["datasets"] = datasets
        return datasets
    except Exception as e:
        app.logger.error(f"dataset scan failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# ✅ load_shapefile_dataset (major optimized)
# ─────────────────────────────────────────────────────────────
def load_shapefile_dataset(dataset_name: str, use_tiles: bool = False) -> List[dict]:
    sig = _dataset_signature(dataset_name, use_tiles)
    cache_key = f"{dataset_name}_{int(bool(use_tiles))}"

    if cache_key in dataset_cache and dataset_cache_signature.get(cache_key) == sig:
        app.logger.info(f"[CACHE HIT] dataset={dataset_name} use_tiles={use_tiles} points={len(dataset_cache[cache_key])}")
        return dataset_cache[cache_key]

    point_shp = os.path.join(get_absolute_path(POINT_FOLDER), f"{dataset_name}.shp")
    if not os.path.exists(point_shp):
        raise FileNotFoundError(f"Point Shapefile not found: {point_shp}")

    pts = load_points_with_pyshp(point_shp, dataset_name)
    if not pts:
        raise ValueError(f"Point Shapefile read failed: {point_shp}")

    idx = get_image_index(dataset_name, force=False)
    all_images = idx["all_images"]
    all_images_set = idx["all_images_set"]
    basename_to_image = idx["basename_to_image"]
    is_static_parent = idx["is_static_parent"]
    use_dataset_in_path = idx["use_dataset_in_path"]

    csv_angles = {}
    csv_angle_adds = {}
    csv_path = os.path.join(get_absolute_path(MATCH_FOLDER), f"{dataset_name}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                image_filename_col = None
                angle_col = None
                angle_add_col = None
                for i, col in enumerate(header):
                    c = (col or "").lower()
                    if "image_filename" in c:
                        image_filename_col = i
                    elif "angle_add" in c:
                        angle_add_col = i
                    elif "angle" in c:
                        angle_col = i

                if image_filename_col is not None:
                    for row in reader:
                        if len(row) <= image_filename_col:
                            continue
                        fn = row[image_filename_col]
                        if angle_col is not None and len(row) > angle_col:
                            try:
                                csv_angles[fn] = float(row[angle_col])
                            except Exception:
                                pass
                        if angle_add_col is not None and len(row) > angle_add_col:
                            try:
                                csv_angle_adds[fn] = float(row[angle_add_col])
                            except Exception:
                                pass
        except Exception as e:
            app.logger.warning(f"csv preload failed: {e}")

    import urllib.parse as _urlparse

    for i, p in enumerate(pts):
        p["dataset_name"] = dataset_name
        p["data_type"] = "shapefile"
        p["frame_index"] = i

        image_filename = p.get("image_file") or p.get("image_filename") or ""
        matched_image = None

        if image_filename and all_images:
            if image_filename in all_images_set:
                matched_image = image_filename
            else:
                bn = os.path.basename(image_filename).lower()
                cand = basename_to_image.get(bn)
                if cand:
                    matched_image = cand[0]

        if matched_image:
            mi = matched_image.replace("\\", "/")
            if is_static_parent and use_dataset_in_path:
                p["image"] = f"/static/img/{dataset_name}/{mi}"
            elif is_static_parent:
                p["image"] = f"/static/img/{mi}"
            else:
                p["image"] = f"/api/img/{dataset_name}/{_urlparse.quote(mi)}"
            p["image_filename"] = matched_image

            img_name = os.path.splitext(matched_image)[0]
            if use_tiles and _has_tiles(dataset_name, img_name):
                p["tile_base_url"] = f"/api/tile/{dataset_name}/{img_name}"
            else:
                p["tile_base_url"] = ""
        else:
            p["image"] = ""
            p["image_filename"] = image_filename or ""
            p["tile_base_url"] = ""

        track_seg_point_id = p.get("track_seg_point_id")
        if not track_seg_point_id:
            if image_filename:
                try:
                    base = os.path.basename(image_filename)
                    name_no_ext, _ = os.path.splitext(base)
                    track_seg_point_id = name_no_ext.rsplit("_", 1)[-1] if "_" in name_no_ext else name_no_ext
                except Exception:
                    track_seg_point_id = f"node_{i}"
            else:
                track_seg_point_id = f"node_{i}"
        p["track_seg_point_id"] = str(track_seg_point_id).strip()

        p["csv_angle"] = csv_angles.get(image_filename, 0.0)
        p["csv_angle_add"] = csv_angle_adds.get(image_filename, 0.0)
        p["final_angle"] = p["csv_angle"] + p["csv_angle_add"]

    dataset_cache[cache_key] = pts
    dataset_cache_signature[cache_key] = sig
    app.logger.info(f"[LOAD DONE] dataset={dataset_name} use_tiles={use_tiles} points={len(pts)} images_index={len(all_images)}")
    return pts


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # points는 서버에만 두고, 페이지에는 전달하지 않음
    try:
        target_id = request.cookies.get("target_node_id")
    except Exception:
        target_id = None

    if target_id:
        target_id = str(target_id).strip()
        found_point = None
        found_index = -1

        # 1) 현재 ACTIVE points에서 찾기
        render_points = _get_active_points_snapshot()
        if render_points:
            for i, p in enumerate(render_points):
                pid = str(p.get("track_seg_point_id", f"node_{i}")).strip()
                if pid == target_id:
                    found_point = p
                    found_index = i
                    break

        # 2) 없으면 node_id -> dataset 찾아서 ACTIVE를 그 dataset으로 세팅 후 재탐색
        if found_point is None:
            ds_name = resolve_dataset_by_node_id(target_id)
            if ds_name:
                try:
                    ds_points = load_shapefile_dataset(ds_name, use_tiles=True)
                    _set_active(ds_points, [ds_name], use_tiles=True)
                    for i, p in enumerate(ds_points):
                        pid = str(p.get("track_seg_point_id", f"node_{i}")).strip()
                        if pid == target_id:
                            found_point = p
                            found_index = i
                            break
                except Exception as e:
                    app.logger.warning(f"[INDEX] target auto-load failed: {e}")

        resp = make_response(
            render_template(
                "index.html",
                points=[],
                target_point=found_point,
                target_index=found_index,
                track_seg_point_id=target_id,
                hide_menu=True,
            )
        )
        try:
            resp.delete_cookie("target_node_id", path="/")
        except Exception:
            pass
        return resp

    return make_response(render_template("index.html", points=[], hide_menu=True))


@app.get("/api/points")
def api_points():
    bbox = request.args.get("bbox", "").strip()
    limit = int(request.args.get("limit", "5000"))

    if not bbox:
        return jsonify([])

    try:
        west, south, east, north = map(float, bbox.split(","))
    except Exception:
        return jsonify([])

    points_by_lat, point_lats = _get_active_points_by_lat_snapshot()
    if not points_by_lat or not point_lats:
        return jsonify([])

    start_idx = bisect_left(point_lats, south)
    end_idx = bisect_right(point_lats, north)

    out = []
    for _lat, lon, p in points_by_lat[start_idx:end_idx]:
        if west <= lon <= east:
            out.append(p)
            if len(out) >= limit:
                break

    return jsonify(out)


@app.route("/pano/admin")
@app.route("/pano/admin/<track_seg_point_id>")
def pano_admin(track_seg_point_id=None):
    # track_seg_point_id가 있으면 해당 노드 정보 전달
    target_point = None
    target_index = -1
    if track_seg_point_id:
        track_seg_point_id = urllib.parse.unquote(str(track_seg_point_id)).strip()
        ds_name = resolve_dataset_by_node_id(track_seg_point_id)
        if ds_name:
            try:
                ds_points = load_shapefile_dataset(ds_name, use_tiles=True)
                for i, p in enumerate(ds_points):
                    pid = str(p.get("track_seg_point_id", f"node_{i}")).strip()
                    if pid == track_seg_point_id:
                        target_point = p
                        target_index = i
                        break
            except Exception as e:
                app.logger.warning(f"[pano/admin] 노드 로드 실패: {e}")

    return make_response(render_template(
        "index.html",
        points=[],
        target_point=target_point,
        target_index=target_index,
        track_seg_point_id=track_seg_point_id or "",
        initialTargetNodeId=track_seg_point_id or "",
        hide_menu=False
    ))


@app.route("/pano/points")
def points():
    bbox = request.args.get("bbox", "").strip()
    limit = int(request.args.get("limit", "2000"))
    include_all = request.args.get("full", "0") in {"1", "true", "True"}

    if include_all:
        return make_response(jsonify(_get_active_points_snapshot()))

    if bbox:
        return make_response(jsonify(_query_active_points_in_bbox(bbox, limit)))

    points_snapshot = _get_active_points_snapshot()
    return make_response(jsonify(points_snapshot[:max(1, limit)]))


@app.route("/api/datasets")
def api_datasets():
    try:
        shapefile_datasets = get_shapefile_datasets()
        return jsonify({"success": True, "datasets": shapefile_datasets, "total_count": len(shapefile_datasets)})
    except Exception as e:
        return jsonify({"error": f"데이터셋 목록 조회 실패: {str(e)}"}), 500


@app.route("/api/load-datasets", methods=["POST"])
def api_load_datasets():
    """
    - points 결과를 gzip 파일 캐시로 저장/재사용
    - 응답은 gzip 압축해서 전송
    - ✅ 전역 ACTIVE_STATE에 저장(모든 클라이언트 공유)
    """
    payload = request.get_json(silent=True) or {}
    req_names = payload.get("datasets", [])
    use_tiles = bool(payload.get("use_tiles", False))

    if not req_names or not isinstance(req_names, list):
        return jsonify({"error": "datasets 배열이 필요합니다."}), 400

    cache_path = _points_cache_path(req_names, use_tiles)
    cached = _read_gzip_json(cache_path)
    if cached and cached.get("success") and isinstance(cached.get("points"), list):
        pts_cached = cached["points"]
        _set_active(pts_cached, req_names, use_tiles)

        resp = _gzip_response_json(cached, 200)
        return resp

    merged = []
    for name in req_names:
        ds_points = load_shapefile_dataset(name, use_tiles=use_tiles)
        merged.extend(ds_points)

    _set_active(merged, req_names, use_tiles)

    resp_payload = {
        "success": True,
        "datasets": req_names,
        "use_tiles": use_tiles,
        "sig_hash": _datasets_sig_hash(req_names, use_tiles),
        "point_count": len(merged),
        "points": merged,
    }

    _write_gzip_json(cache_path, resp_payload)

    resp = _gzip_response_json(resp_payload, 200)
    return resp


@app.route("/api/clear-cache", methods=["POST"])
def api_clear_cache():
    global dataset_cache, dataset_cache_signature, image_index_cache, tile_root_map, tile_root_map_sig, dataset_list_cache

    dataset_cache.clear()
    dataset_cache_signature.clear()
    image_index_cache.clear()
    dataset_list_cache["sig"] = None
    dataset_list_cache["datasets"] = []
    tile_root_map = {}
    tile_root_map_sig = None

    # ACTIVE도 초기화
    _set_active([], [], use_tiles=True)

    removed = 0
    cache_dir = get_absolute_path(CACHE_FOLDER)
    try:
        for fn in os.listdir(cache_dir):
            if fn.startswith("points_") and fn.endswith(".json.gz"):
                try:
                    os.remove(os.path.join(cache_dir, fn))
                    removed += 1
                except Exception:
                    pass
        try:
            nif = get_node_index_file()
            if os.path.exists(nif):
                os.remove(nif)
        except Exception:
            pass
    except Exception:
        pass

    return jsonify({"success": True, "message": "캐시가 클리어되었습니다.", "removed_files": removed})


@app.route("/api/folder-settings", methods=["GET"])
def api_get_folder_settings():
    current_settings = {
        "img_folder": normalize_path_for_storage(UPLOAD_IMG_FOLDER),
        "track_folder": normalize_path_for_storage(TRACK_FOLDER),
        "point_folder": normalize_path_for_storage(POINT_FOLDER),
        "match_folder": normalize_path_for_storage(MATCH_FOLDER),
        "tile_folder": normalize_path_for_storage(TILE_FOLDER),
    }
    return jsonify(current_settings)


@app.route("/api/folder-settings", methods=["POST"])
def api_set_folder_settings():
    global folder_settings, UPLOAD_IMG_FOLDER, TRACK_FOLDER, POINT_FOLDER, MATCH_FOLDER, TILE_FOLDER
    global dataset_cache, dataset_cache_signature, image_index_cache, tile_root_map, tile_root_map_sig

    try:
        payload = request.get_json(silent=True) or {}

        def norm(p: str) -> str:
            return normalize_path_for_storage(p.strip()) if p and p.strip() else ""

        new_img = norm(payload.get("img_folder", "")) or folder_settings.get("img_folder", UPLOAD_IMG_FOLDER)
        new_trk = norm(payload.get("track_folder", "")) or folder_settings.get("track_folder", TRACK_FOLDER)
        new_pt = norm(payload.get("point_folder", "")) or folder_settings.get("point_folder", POINT_FOLDER)
        new_mch = norm(payload.get("match_folder", "")) or folder_settings.get("match_folder", MATCH_FOLDER)
        new_tile = norm(payload.get("tile_folder", "")) or folder_settings.get("tile_folder", TILE_FOLDER)

        if not os.path.exists(get_absolute_path(new_img)):
            return jsonify({"error": f"이미지 폴더가 존재하지 않습니다: {new_img}"}), 400

        try:
            os.makedirs(get_absolute_path(new_tile), exist_ok=True)
        except Exception:
            pass

        folder_settings = {
            "img_folder": new_img,
            "track_folder": new_trk,
            "point_folder": new_pt,
            "match_folder": new_mch,
            "tile_folder": new_tile,
        }

        UPLOAD_IMG_FOLDER = new_img
        TRACK_FOLDER = new_trk
        POINT_FOLDER = new_pt
        MATCH_FOLDER = new_mch
        TILE_FOLDER = new_tile

        dataset_cache.clear()
        dataset_cache_signature.clear()
        image_index_cache.clear()
        tile_root_map = {}
        tile_root_map_sig = None

        # 폴더 바뀌면 ACTIVE도 비우는 게 안전
        _set_active([], [], use_tiles=True)

        return jsonify({"success": True, "message": "폴더 경로가 업데이트되었습니다.", "settings": folder_settings})
    except Exception as e:
        return jsonify({"error": f"폴더 경로 설정 실패: {str(e)}"}), 500


@app.route("/api/track-line/<dataset_name>")
def api_get_track_line(dataset_name):
    try:
        track_folder_abs = get_absolute_path(TRACK_FOLDER)
        track_shp = os.path.join(track_folder_abs, f"{dataset_name}.shp")

        if not os.path.exists(track_folder_abs):
            return jsonify(
                {"success": True, "lines": [], "dataset_name": dataset_name, "count": 0, "warning": f"TRACK_FOLDER가 존재하지 않습니다: {TRACK_FOLDER}"}
            ), 200

        if not os.path.exists(track_shp):
            return jsonify(
                {"success": True, "lines": [], "dataset_name": dataset_name, "count": 0, "error": f"Track Shapefile을 찾을 수 없습니다: {track_shp}"}
            ), 200

        lines = load_shapefile_with_pyshp(track_shp)
        return jsonify({"success": True, "lines": lines, "dataset_name": dataset_name, "count": len(lines)})
    except Exception as e:
        return jsonify({"error": f"Track 라인 로드 실패: {str(e)}"}), 500


@app.route("/api/img/<dataset_name>/<path:image_path>")
def api_get_image(dataset_name, image_path):
    try:
        decoded = urllib.parse.unquote(image_path)
        img_folder_for_ds = get_image_folder_path(dataset_name)
        actual = os.path.normpath(os.path.join(img_folder_for_ds, decoded))
        base = os.path.normpath(img_folder_for_ds)
        if not actual.startswith(base):
            return jsonify({"error": "Invalid path"}), 403
        if not (os.path.exists(actual) and os.path.isfile(actual)):
            return jsonify({"error": "Image not found"}), 404

        ext = os.path.splitext(actual)[1].lower()
        mimetype = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        return send_file(actual, mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": f"Image serving error: {str(e)}"}), 500


@app.route("/api/tile/<dataset_name>/<img_name>/<int:level>/<int:y>/<int:x>.jpg")
def api_get_tile(dataset_name, img_name, level, y, x):
    try:
        base_dir = _resolve_tile_base_dir(dataset_name, img_name)
        if not base_dir:
            return ("", 404)

        tile_path = os.path.join(base_dir, str(level), str(y), f"{x}.jpg")
        if os.path.exists(tile_path):
            return send_file(tile_path, mimetype="image/jpeg")
        return ("", 404)
    except Exception as e:
        app.logger.error(f"[tile] error: {e}")
        return ("", 500)


@app.route("/api/tile-composite/<dataset_name>/<img_name>")
def api_tile_composite(dataset_name, img_name):
    """
    타일이 없을 때 전체 이미지를 레벨별로 리사이즈하여 제공
    level 파라미터: 0 (원본), 1 (1/2), 2 (1/4)
    """
    try:
        level = int(request.args.get("level", "0"))

        # 이미지 파일 찾기
        img_folder = get_image_folder_path(dataset_name)
        img_candidates = [
            os.path.join(img_folder, f"{img_name}.jpg"),
            os.path.join(img_folder, f"{img_name}.jpeg"),
            os.path.join(img_folder, f"{img_name}.png"),
        ]

        img_path = None
        for candidate in img_candidates:
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            return ("", 404)

        # 레벨에 따른 리사이즈
        if PILLOW_AVAILABLE and Image:
            img = Image.open(img_path)
            width, height = img.size

            if level == 2:
                new_size = (width // 4, height // 4)
            elif level == 1:
                new_size = (width // 2, height // 2)
            else:
                new_size = (width, height)

            if new_size != (width, height):
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            from io import BytesIO
            output = BytesIO()
            img.save(output, format="JPEG", quality=85)
            output.seek(0)
            return send_file(output, mimetype="image/jpeg")
        else:
            return send_file(img_path, mimetype="image/jpeg")
    except Exception as e:
        app.logger.error(f"[tile-composite] error: {e}")
        return ("", 500)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/pano/<track_seg_point_id>")
def view_node(track_seg_point_id):
    track_seg_point_id = urllib.parse.unquote(str(track_seg_point_id)).strip()

    # 1) ACTIVE에 target이 없으면 node_id로 dataset 찾아서 ACTIVE를 로딩
    def _has_target(points, tid):
        for i, p in enumerate(points or []):
            pid = str(p.get("track_seg_point_id", f"node_{i}")).strip()
            if pid == tid:
                return True
        return False

    pts = _get_active_points_snapshot()
    if not _has_target(pts, track_seg_point_id):
        ds_name = resolve_dataset_by_node_id(track_seg_point_id)
        if ds_name:
            try:
                ds_points = load_shapefile_dataset(ds_name, use_tiles=True)
                _set_active(ds_points, [ds_name], use_tiles=True)
                app.logger.info(f"[PANO] auto-loaded dataset by node_id={track_seg_point_id}: {ds_name} pts={len(ds_points)}")
            except Exception as e:
                app.logger.warning(f"[PANO] auto-load failed: {e}")

    return make_response(render_template(
        "index.html",
        points=[],
        track_seg_point_id=track_seg_point_id,
        initialTargetNodeId=track_seg_point_id,
        hide_menu=True
    ))


@app.route("/api/build-node-index", methods=["POST"])
def api_build_node_index():
    data = build_node_index(force=True)
    return jsonify({"success": True, "sig": data.get("sig"), "count": len(data.get("map", {}))})


def warmup_indexes():
    try:
        build_node_index(force=False)
        _build_tile_root_map(force=False)

        # ✅ default dataset preload -> ACTIVE_STATE로 세팅
        if DEFAULT_DATASETS:
            names = [x.strip() for x in DEFAULT_DATASETS.split(",") if x.strip()]
            merged = []
            for name in names:
                merged.extend(load_shapefile_dataset(name, use_tiles=DEFAULT_USE_TILES))
            _set_active(merged, names, use_tiles=DEFAULT_USE_TILES)
            app.logger.info(f"[WARMUP] default loaded: {','.join(names)} points={len(merged)}")
        else:
            # 환경변수 없으면 첫 dataset을 default로
            ds = get_shapefile_datasets()
            if ds:
                name = ds[0]["name"]
                pts = load_shapefile_dataset(name, use_tiles=DEFAULT_USE_TILES)
                _set_active(pts, [name], use_tiles=DEFAULT_USE_TILES)
                app.logger.info(f"[WARMUP] default loaded(first): {name} points={len(pts)}")

        app.logger.info("[WARMUP] done")
    except Exception as e:
        app.logger.warning(f"[WARMUP] failed: {e}")


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        warmup_indexes()
        from waitress import serve
        threads = int(os.getenv("THREADS", "8"))
        serve(app, host="127.0.0.1", port="5000", threads=threads)
    else:
        raise SystemExit(
            "운영은 gunicorn으로 실행하세요: "
            "nohup gunicorn app:app -b 127.0.0.1:5000 -w 2 --threads 8 --timeout 120 "
            "--config gunicorn.conf.py > gunicorn.log 2>&1 &"
        )
