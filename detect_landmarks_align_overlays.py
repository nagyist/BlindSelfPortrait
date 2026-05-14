#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageChops, ImageDraw, ImageOps, UnidentifiedImageError


PROMPT = """Task: Point to eyes, bottom of nose, lips, top of head, and bottom of chin.
Return a JSON array of objects.
Each object must have:
- "point": [y, x] (coordinates 0-1000)
- "label": text label
Example: [{"point": [500, 500], "label": "example"}]
Return ONLY the JSON."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect face landmarks on source faces and generated outlines, estimate "
            "scale/translation transforms, and render aligned multiply overlays."
        )
    )
    parser.add_argument("--outline", default="outline")
    parser.add_argument("--faces", default="faces")
    parser.add_argument("--output", default="overlaid-aligned")
    parser.add_argument("--landmarks", default="landmarks")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="Ignore cached Gemini JSON.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write landmark visualization images in landmarks/debug/.",
    )
    parser.add_argument(
        "--draw-match-points",
        action="store_true",
        help=(
            "Draw transformed face landmarks in red and outline landmarks in blue "
            "on the aligned overlay PNGs."
        ),
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
            if raw_line.strip().split("=", 1)[1].strip().startswith('"'):
                value = bytes(value, "utf-8").decode("unicode_escape")

        os.environ[key] = value


def filename_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in value)


def outline_pngs(outline_dir: Path) -> list[Path]:
    return sorted(path for path in outline_dir.glob("*.png") if path.is_file())


def source_from_json(outline_path: Path) -> Path | None:
    json_path = outline_path.with_suffix(".json")
    if not json_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    source = payload.get("request", {}).get("source_image")
    if not isinstance(source, str) or not source:
        return None
    return Path(source)


def source_from_filename(outline_path: Path, faces_dir: Path) -> Path | None:
    marker = "__gpt-image-2__"
    if marker not in outline_path.stem:
        return None
    face_stem = outline_path.stem.split(marker, 1)[0]
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = faces_dir / f"{face_stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def source_face(outline_path: Path, faces_dir: Path) -> Path | None:
    source = source_from_json(outline_path)
    if source is not None and source.is_file():
        return source
    return source_from_filename(outline_path, faces_dir)


def build_pairs(outline_dir: Path, faces_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for outline_path in outline_pngs(outline_dir):
        face_path = source_face(outline_path, faces_dir)
        if face_path is None:
            continue
        pairs.append(
            {
                "id": outline_path.stem,
                "outline_path": outline_path,
                "face_path": face_path,
            }
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def image_to_png_base64(path: Path) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), image.size


def gemini_request_payload(
    image_base64: str,
    model: str,
    temperature: float,
    redact: bool = False,
) -> dict[str, Any]:
    data = "<BASE64_IMAGE_DATA_REDACTED>" if redact else image_base64
    return {
        "model": model,
        "contents": {
            "parts": [
                {
                    "inlineData": {
                        "data": data,
                        "mimeType": "image/png",
                    }
                },
                {"text": PROMPT},
            ]
        },
        "config": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


def gemini_rest_payload(image_base64: str, temperature: float) -> dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "data": image_base64,
                            "mimeType": "image/png",
                        }
                    },
                    {"text": PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


def generate_content(
    api_key: str,
    image_path: Path,
    model: str,
    temperature: float,
    timeout: int = 120,
) -> dict[str, Any]:
    image_base64, image_size = image_to_png_base64(image_path)
    body = json.dumps(gemini_rest_payload(image_base64, temperature)).encode("utf-8")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
            headers = dict(response.headers.items())
    except HTTPError as exc:
        raw = exc.read()
        status = exc.code
        headers = dict(exc.headers.items())
    duration = time.perf_counter() - started

    try:
        response_json = json.loads(raw)
    except json.JSONDecodeError:
        response_json = {"raw_text": raw.decode("utf-8", errors="replace")}

    return {
        "request": gemini_request_payload(
            "<BASE64_IMAGE_DATA_REDACTED>",
            model,
            temperature,
            redact=True,
        ),
        "image_path": str(image_path),
        "image_size": list(image_size),
        "started_at": started_at.isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 3),
        "status": status,
        "headers": {
            key: value
            for key, value in headers.items()
            if key.lower().startswith(("x-", "content-type"))
        },
        "response": response_json,
    }


def response_text(response_json: dict[str, Any]) -> str:
    if isinstance(response_json.get("raw_text"), str):
        return response_json["raw_text"]
    candidates = response_json.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return json.dumps(response_json)
    parts = candidates[0].get("content", {}).get("parts", [])
    if not isinstance(parts, list):
        return json.dumps(response_json)
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text for text in texts if isinstance(text, str))


def parse_jsonish(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[[\s\S]*\]", stripped)
    if array_match:
        return json.loads(array_match.group(0))
    object_match = re.search(r"\{[\s\S]*\}", stripped)
    if object_match:
        return json.loads(object_match.group(0))
    raise ValueError("Could not parse JSON landmarks.")


def raw_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("items", "points", "landmarks", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        for value in parsed.values():
            if isinstance(value, list):
                return value
    return []


def label_category(label: str) -> str | None:
    normalized = label.lower().strip()
    if "eye" in normalized or "pupil" in normalized:
        return "eye"
    if "top" in normalized and any(
        token in normalized for token in ("head", "hair", "skull", "forehead")
    ):
        return "top_head"
    if "crown" in normalized:
        return "top_head"
    if "chin" in normalized:
        return "bottom_chin"
    if "jaw" in normalized and "bottom" in normalized:
        return "bottom_chin"
    if "nose" in normalized:
        return "nose"
    if "lip" in normalized or "mouth" in normalized:
        return "lips"
    return None


def parse_points(landmark_json: dict[str, Any]) -> dict[str, Any]:
    text = response_text(landmark_json["response"])
    parsed = parse_jsonish(text)
    points: list[dict[str, Any]] = []
    for item in raw_items(parsed):
        if not isinstance(item, dict):
            continue
        point = item.get("point") or item.get("point_2d") or item.get("coordinates")
        label = item.get("label", "")
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            y = float(point[0])
            x = float(point[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            continue
        category = label_category(str(label))
        if category is None:
            continue
        points.append(
            {
                "x": x,
                "y": y,
                "label": str(label),
                "category": category,
            }
        )
    return {"raw_parsed": parsed, "points": points, "anchors": anchors_from_points(points)}


def choose_eye_pair(eyes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(eyes) <= 2:
        return sorted(eyes, key=lambda point: point["x"])
    best_pair: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for i, first in enumerate(eyes):
        for second in eyes[i + 1 :]:
            left, right = sorted([first, second], key=lambda point: point["x"])
            x_sep = right["x"] - left["x"]
            y_delta = abs(right["y"] - left["y"])
            center_y = (left["y"] + right["y"]) / 2
            center_x = (left["x"] + right["x"]) / 2
            score = x_sep - 2.0 * y_delta
            if x_sep < 70:
                score -= 250
            if x_sep > 450:
                score -= 150
            if not (220 <= center_y <= 620):
                score -= 150
            if not (250 <= center_x <= 750):
                score -= 75
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, left, right)
    if best_pair is None:
        return sorted(eyes, key=lambda point: point["x"])[:2]
    return [best_pair[1], best_pair[2]]


def median_point(points: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "x": statistics.median(point["x"] for point in points),
        "y": statistics.median(point["y"] for point in points),
    }


def anchors_from_points(points: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    anchors: dict[str, dict[str, float]] = {}
    eyes = [point for point in points if point["category"] == "eye"]
    top_heads = [point for point in points if point["category"] == "top_head"]
    chins = [point for point in points if point["category"] == "bottom_chin"]
    noses = [point for point in points if point["category"] == "nose"]
    lips = [point for point in points if point["category"] == "lips"]

    if eyes:
        eye_pair = choose_eye_pair(eyes)
        if len(eye_pair) == 2:
            anchors["left_eye"] = {"x": eye_pair[0]["x"], "y": eye_pair[0]["y"]}
            anchors["right_eye"] = {"x": eye_pair[1]["x"], "y": eye_pair[1]["y"]}
            anchors["eye_center"] = {
                "x": (eye_pair[0]["x"] + eye_pair[1]["x"]) / 2,
                "y": (eye_pair[0]["y"] + eye_pair[1]["y"]) / 2,
            }
        elif len(eye_pair) == 1:
            center_x = statistics.median(
                [point["x"] for point in noses + lips] or [500.0]
            )
            side = "left_eye" if eye_pair[0]["x"] < center_x else "right_eye"
            anchors[side] = {"x": eye_pair[0]["x"], "y": eye_pair[0]["y"]}
            anchors["eye_center"] = {"x": eye_pair[0]["x"], "y": eye_pair[0]["y"]}

    if top_heads:
        top_head = min(top_heads, key=lambda point: point["y"])
        anchors["top_head"] = {"x": top_head["x"], "y": top_head["y"]}

    if chins:
        bottom_chin = max(chins, key=lambda point: point["y"])
        anchors["bottom_chin"] = {"x": bottom_chin["x"], "y": bottom_chin["y"]}

    if noses:
        bottom_labeled = [
            point for point in noses if "bottom" in str(point.get("label", "")).lower()
        ]
        candidates = bottom_labeled or noses
        nose = max(candidates, key=lambda point: point["y"])
        anchors["bottom_of_nose"] = {"x": nose["x"], "y": nose["y"]}

    if lips:
        anchors["lips"] = median_point(lips)

    return anchors


def normalized_to_pixels(
    point: dict[str, float],
    size: tuple[int, int],
) -> tuple[float, float]:
    width, height = size
    return (point["x"] / 1000.0 * width, point["y"] / 1000.0 * height)


def head_height_pixels(
    anchors: dict[str, dict[str, float]],
    size: tuple[int, int],
) -> float | None:
    if "top_head" not in anchors or "bottom_chin" not in anchors:
        return None
    top = normalized_to_pixels(anchors["top_head"], size)
    chin = normalized_to_pixels(anchors["bottom_chin"], size)
    height = chin[1] - top[1]
    if height <= size[1] * 0.15:
        return None
    return height


def estimate_transform(
    face_landmarks: dict[str, Any],
    outline_landmarks: dict[str, Any],
    face_size: tuple[int, int],
    outline_size: tuple[int, int],
) -> dict[str, Any]:
    face_anchors = face_landmarks.get("anchors", {})
    outline_anchors = outline_landmarks.get("anchors", {})
    anchor_names = [
        "top_head",
        "left_eye",
        "right_eye",
        "eye_center",
        "bottom_of_nose",
        "lips",
        "bottom_chin",
    ]
    weights = {
        "top_head": 0.65,
        "left_eye": 1.3,
        "right_eye": 1.3,
        "eye_center": 1.0,
        "bottom_of_nose": 0.8,
        "lips": 0.8,
        "bottom_chin": 0.75,
    }

    scale = 1.0
    scale_candidates = []
    face_head_height = head_height_pixels(face_anchors, face_size)
    outline_head_height = head_height_pixels(outline_anchors, outline_size)
    if face_head_height is not None and outline_head_height is not None:
        candidate_scale = outline_head_height / face_head_height
        scale_candidates.append(
            {
                "source": "top_head_to_bottom_chin",
                "scale": candidate_scale,
                "face_distance": face_head_height,
                "outline_distance": outline_head_height,
                "accepted": 0.6 <= candidate_scale <= 1.6,
            }
        )
        if 0.6 <= candidate_scale <= 1.6:
            scale = candidate_scale

    candidates = []
    for name in anchor_names:
        if name not in face_anchors or name not in outline_anchors:
            continue
        face_xy = normalized_to_pixels(face_anchors[name], face_size)
        outline_xy = normalized_to_pixels(outline_anchors[name], outline_size)
        dx = outline_xy[0] - scale * face_xy[0]
        dy = outline_xy[1] - scale * face_xy[1]
        if abs(dx) > face_size[0] * 0.7 or abs(dy) > face_size[1] * 0.7:
            continue
        candidates.append(
            {
                "anchor": name,
                "dx": dx,
                "dy": dy,
                "weight": weights[name],
                "scale": scale,
                "face_xy": list(face_xy),
                "outline_xy": list(outline_xy),
            }
        )

    if not candidates:
        return {
            "scale": round(scale, 5),
            "dx": 0.0,
            "dy": 0.0,
            "scale_candidates": scale_candidates,
            "used": [],
            "rejected": [],
            "status": "no_common_landmarks",
        }

    median_dx = statistics.median(candidate["dx"] for candidate in candidates)
    median_dy = statistics.median(candidate["dy"] for candidate in candidates)
    used = []
    rejected = []
    for candidate in candidates:
        distance = math.hypot(candidate["dx"] - median_dx, candidate["dy"] - median_dy)
        if len(candidates) >= 3 and distance > 120:
            rejected.append(candidate)
        else:
            used.append(candidate)
    if not used:
        used = candidates
        rejected = []

    weight_sum = sum(candidate["weight"] for candidate in used)
    dx = sum(candidate["dx"] * candidate["weight"] for candidate in used) / weight_sum
    dy = sum(candidate["dy"] * candidate["weight"] for candidate in used) / weight_sum

    return {
        "scale": round(scale, 5),
        "dx": round(dx, 3),
        "dy": round(dy, 3),
        "scale_candidates": [
            {
                **candidate,
                "scale": round(candidate["scale"], 5),
                "face_distance": round(candidate["face_distance"], 3),
                "outline_distance": round(candidate["outline_distance"], 3),
            }
            for candidate in scale_candidates
        ],
        "used": [
            {
                **candidate,
                "dx": round(candidate["dx"], 3),
                "dy": round(candidate["dy"], 3),
                "scale": round(candidate["scale"], 5),
            }
            for candidate in used
        ],
        "rejected": [
            {
                **candidate,
                "dx": round(candidate["dx"], 3),
                "dy": round(candidate["dy"], 3),
                "scale": round(candidate["scale"], 5),
            }
            for candidate in rejected
        ],
        "status": "ok",
    }


def transform_face(face: Image.Image, scale: float, dx: float, dy: float) -> Image.Image:
    if scale <= 0:
        scale = 1.0
    return face.transform(
        face.size,
        Image.Transform.AFFINE,
        (1 / scale, 0, -dx / scale, 0, 1 / scale, -dy / scale),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def transformed_landmark_xy(
    point: dict[str, Any],
    size: tuple[int, int],
    scale: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    x, y = normalized_to_pixels(point, size)
    return (scale * x + dx, scale * y + dy)


def draw_match_points(
    image: Image.Image,
    face_landmarks: dict[str, Any],
    outline_landmarks: dict[str, Any],
    scale: float,
    dx: float,
    dy: float,
) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for point in outline_landmarks.get("points", []):
        x, y = normalized_to_pixels(point, image.size)
        radius = 12
        draw.ellipse(
            (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
            outline=(255, 255, 255),
            width=5,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(20, 90, 255),
            width=4,
        )

    for point in face_landmarks.get("points", []):
        x, y = transformed_landmark_xy(point, (width, height), scale, dx, dy)
        radius = 7
        draw.ellipse(
            (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
            fill=(255, 255, 255),
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(235, 30, 35),
        )


def render_aligned_overlay(
    face_path: Path,
    outline_path: Path,
    output_path: Path,
    scale: float,
    dx: float,
    dy: float,
    face_landmarks: dict[str, Any] | None = None,
    outline_landmarks: dict[str, Any] | None = None,
    draw_points: bool = False,
) -> None:
    with Image.open(face_path) as face_loaded, Image.open(outline_path) as outline_loaded:
        face = ImageOps.exif_transpose(face_loaded).convert("RGB")
        outline = ImageOps.exif_transpose(outline_loaded).convert("RGB")
    if outline.size != face.size:
        outline = outline.resize(face.size, Image.Resampling.LANCZOS)
    shifted_face = transform_face(face, scale, dx, dy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = ImageChops.multiply(shifted_face, outline)
    if draw_points and face_landmarks is not None and outline_landmarks is not None:
        draw_match_points(overlay, face_landmarks, outline_landmarks, scale, dx, dy)
    overlay.save(output_path, "PNG")


def draw_debug(
    image_path: Path,
    landmarks: dict[str, Any],
    output_path: Path,
) -> None:
    with Image.open(image_path) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    colors = {
        "eye": (40, 110, 255),
        "top_head": (30, 180, 90),
        "bottom_chin": (150, 80, 255),
        "nose": (255, 110, 40),
        "lips": (230, 40, 180),
    }
    for point in landmarks.get("points", []):
        x = point["x"] / 1000 * width
        y = point["y"] / 1000 * height
        color = colors.get(point["category"], (0, 0, 0))
        radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.text((x + 10, y - 8), point["label"], fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")


def landmark_json_path(landmarks_dir: Path, kind: str, pair_id: str) -> Path:
    return landmarks_dir / kind / f"{filename_token(pair_id)}.json"


def load_or_detect(
    api_key: str,
    image_path: Path,
    pair_id: str,
    kind: str,
    landmarks_dir: Path,
    model: str,
    temperature: float,
    force: bool,
) -> dict[str, Any]:
    output_path = landmark_json_path(landmarks_dir, kind, pair_id)
    if output_path.is_file() and not force:
        return json.loads(output_path.read_text())

    result = generate_content(api_key, image_path, model, temperature)
    result["kind"] = kind
    result["pair_id"] = pair_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def detect_job(
    api_key: str,
    image_path: Path,
    pair_id: str,
    kind: str,
    landmarks_dir: Path,
    model: str,
    temperature: float,
    force: bool,
) -> tuple[tuple[str, str], dict[str, Any]]:
    result = load_or_detect(
        api_key,
        image_path,
        pair_id,
        kind,
        landmarks_dir,
        model,
        temperature,
        force,
    )
    if int(result.get("status", 0)) >= 400:
        raise RuntimeError(f"Gemini HTTP {result.get('status')} for {image_path}")
    parsed = parse_points(result)
    parsed_path = landmarks_dir / "parsed" / kind / f"{filename_token(pair_id)}.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(json.dumps(parsed, indent=2) + "\n")
    return (pair_id, kind), parsed


def main() -> int:
    args = parse_args()
    load_env_file(Path(".env"))
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment or .env", file=sys.stderr)
        return 1

    outline_dir = Path(args.outline)
    faces_dir = Path(args.faces)
    output_dir = Path(args.output)
    landmarks_dir = Path(args.landmarks)
    if not outline_dir.is_dir():
        print(f"Outline directory not found: {outline_dir}", file=sys.stderr)
        return 1
    if not faces_dir.is_dir():
        print(f"Faces directory not found: {faces_dir}", file=sys.stderr)
        return 1

    pairs = build_pairs(outline_dir, faces_dir, args.limit)
    if not pairs:
        print("No image pairs found.", file=sys.stderr)
        return 1

    print(f"pairs={len(pairs)}")
    print(f"workers={args.workers}")
    print(f"landmarks={landmarks_dir}")
    print(f"output={output_dir}")

    jobs = []
    for pair in pairs:
        jobs.append((pair["face_path"], pair["id"], "face"))
        jobs.append((pair["outline_path"], pair["id"], "outline"))

    parsed_by_pair: dict[str, dict[str, Any]] = {
        pair["id"]: {"pair": pair} for pair in pairs
    }
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                detect_job,
                api_key,
                image_path,
                pair_id,
                kind,
                landmarks_dir,
                args.model,
                args.temperature,
                args.force,
            ): (pair_id, kind, image_path)
            for image_path, pair_id, kind in jobs
        }
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                pair_id, kind, image_path = futures.pop(future)
                try:
                    (_, _), parsed = future.result()
                except Exception as exc:
                    print(f"landmark_failed kind={kind} image={image_path} error={exc}", file=sys.stderr)
                    return 1
                parsed_by_pair[pair_id][kind] = parsed
                completed += 1
                print(
                    f"landmarks {completed:03d}/{len(jobs)} kind={kind} "
                    f"points={len(parsed.get('points', []))} image={image_path}"
                )

    rendered = 0
    alignment_dir = landmarks_dir / "alignment"
    for pair_id, bundle in parsed_by_pair.items():
        pair = bundle["pair"]
        face_path = pair["face_path"]
        outline_path = pair["outline_path"]
        try:
            with Image.open(face_path) as face_loaded, Image.open(outline_path) as outline_loaded:
                face_size = ImageOps.exif_transpose(face_loaded).size
                outline_size = ImageOps.exif_transpose(outline_loaded).size
        except (OSError, UnidentifiedImageError) as exc:
            print(f"image_failed pair={pair_id} error={exc}", file=sys.stderr)
            continue

        transform = estimate_transform(
            bundle["face"],
            bundle["outline"],
            face_size,
            outline_size,
        )
        output_path = output_dir / f"{pair_id}.png"
        render_aligned_overlay(
            face_path,
            outline_path,
            output_path,
            transform["scale"],
            transform["dx"],
            transform["dy"],
            bundle["face"],
            bundle["outline"],
            args.draw_match_points,
        )
        alignment_payload = {
            "pair_id": pair_id,
            "face_path": str(face_path),
            "outline_path": str(outline_path),
            "output_path": str(output_path),
            "face_landmarks": bundle["face"],
            "outline_landmarks": bundle["outline"],
            "transform": transform,
            "translation": transform,
            "draw_match_points": args.draw_match_points,
        }
        alignment_dir.mkdir(parents=True, exist_ok=True)
        (alignment_dir / f"{filename_token(pair_id)}.json").write_text(
            json.dumps(alignment_payload, indent=2) + "\n"
        )
        if args.debug:
            draw_debug(
                face_path,
                bundle["face"],
                landmarks_dir / "debug" / f"{filename_token(pair_id)}__face.png",
            )
            draw_debug(
                outline_path,
                bundle["outline"],
                landmarks_dir / "debug" / f"{filename_token(pair_id)}__outline.png",
            )
        rendered += 1
        print(
            f"aligned {rendered:03d}/{len(pairs)} scale={transform['scale']} "
            f"dx={transform['dx']} "
            f"dy={transform['dy']} used={len(transform['used'])} output={output_path}"
        )

    print(f"summary pairs={len(pairs)} rendered={rendered} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
