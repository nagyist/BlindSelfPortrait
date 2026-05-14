#!/usr/bin/env python3
"""Classify extracted face crops as masked/unmasked with Gemini."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PROMPT = """Task: Determine whether the visible person is wearing a face mask or other covering over the mouth and/or nose.

Return ONLY a JSON object with these fields:
- "wearing_mask": true or false
- "confidence": number from 0 to 1
- "reason": brief text

Classify surgical masks, respirators, cloth masks, scarves, hands, or other coverings that cover the mouth or nose as wearing_mask=true.
Classify glasses, hats, shadows, facial hair, and normal clothing that does not cover the mouth or nose as wearing_mask=false.
If uncertain, choose the most likely answer."""


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def response_path_for(image_path: Path, responses_dir: Path) -> Path:
    return responses_dir / f"{image_path.stem}.json"


def images_from_manifest(manifest: Path) -> list[Path]:
    rows: list[Path] = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            output = row.get("output")
            if output:
                rows.append(Path(output))
    return rows


def images_from_args(args: argparse.Namespace) -> list[Path]:
    images: list[Path] = []
    if args.manifest:
        images.extend(images_from_manifest(Path(args.manifest)))
    for path_text in args.images:
        path = Path(path_text)
        if path.is_dir():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                images.extend(sorted(path.glob(pattern)))
        else:
            images.append(path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for image in images:
        resolved = image.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(image)
    if args.name_prefix:
        unique = [image for image in unique if image.name.startswith(args.name_prefix)]
    return unique


def image_to_128_png_b64(path: Path) -> tuple[str, list[int], list[int]]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = [image.width, image.height]
        resized = image.resize((128, 128), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        resized.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), original_size, [128, 128]


def call_gemini(
    *,
    api_key: str,
    model: str,
    image_b64: str,
    timeout: float,
) -> tuple[int | None, dict[str, Any] | str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "data": image_b64,
                            "mimeType": "image/png",
                        }
                    },
                    {"text": PROMPT},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed: dict[str, Any] | str = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return error.code, parsed
    except urllib.error.URLError as error:
        return None, str(error)


def call_gemini_with_retries(
    *,
    api_key: str,
    model: str,
    image_b64: str,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> tuple[int | None, dict[str, Any] | str, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    response: dict[str, Any] | str = ""
    status: int | None = None
    for attempt in range(1, retries + 2):
        attempt_started = time.monotonic()
        status, response = call_gemini(
            api_key=api_key,
            model=model,
            image_b64=image_b64,
            timeout=timeout,
        )
        attempts.append(
            {
                "attempt": attempt,
                "status": status,
                "duration_seconds": time.monotonic() - attempt_started,
            }
        )
        if status is not None and status < 429:
            break
        if status is not None and 400 <= status < 500 and status != 429:
            break
        if attempt <= retries:
            time.sleep(retry_delay * attempt)
    return status, response, attempts


def extract_candidate_text(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return None
    parts_text: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts_text.append(part["text"])
    return "\n".join(parts_text).strip() or None


def parse_classification(response: Any) -> dict[str, Any]:
    text = extract_candidate_text(response)
    parsed: Any = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

    result = {
        "wearing_mask": None,
        "confidence": None,
        "reason": None,
        "raw_text": text,
    }
    if isinstance(parsed, dict):
        wearing_mask = parsed.get("wearing_mask")
        if isinstance(wearing_mask, bool):
            result["wearing_mask"] = wearing_mask
        elif isinstance(wearing_mask, str):
            lowered = wearing_mask.strip().lower()
            if lowered in {"true", "yes", "masked", "wearing_mask"}:
                result["wearing_mask"] = True
            elif lowered in {"false", "no", "unmasked", "not_masked"}:
                result["wearing_mask"] = False

        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)):
            result["confidence"] = float(confidence)
        elif isinstance(confidence, str):
            try:
                result["confidence"] = float(confidence)
            except ValueError:
                pass

        reason = parsed.get("reason")
        if isinstance(reason, str):
            result["reason"] = reason

    if result["wearing_mask"] is None and text:
        lowered = text.lower()
        if '"wearing_mask": true' in lowered or "wearing_mask true" in lowered:
            result["wearing_mask"] = True
        elif '"wearing_mask": false' in lowered or "wearing_mask false" in lowered:
            result["wearing_mask"] = False
    return result


def process_image(
    *,
    image_path: Path,
    responses_dir: Path,
    api_key: str,
    model: str,
    timeout: float,
    delete_masked: bool,
    force: bool,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    output_path = response_path_for(image_path, responses_dir)
    if output_path.exists() and not force:
        saved = json.loads(output_path.read_text())
        classification = saved.get("classification", {})
        deleted = False
        if classification.get("wearing_mask") is True and delete_masked and image_path.exists():
            image_path.unlink()
            deleted = True
        return {
            "image": str(image_path),
            "response_json": str(output_path),
            "status": saved.get("status"),
            "classification": classification,
            "deleted": deleted,
            "cached": True,
        }

    image_b64, original_size, sent_size = image_to_128_png_b64(image_path)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    status, response, attempts = call_gemini_with_retries(
        api_key=api_key,
        model=model,
        image_b64=image_b64,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    duration = time.monotonic() - started
    classification = parse_classification(response)

    record = {
        "source_image": str(image_path),
        "model": model,
        "prompt": PROMPT,
        "started_at": started_at,
        "duration_seconds": duration,
        "original_size": original_size,
        "sent_size": sent_size,
        "sent_mime_type": "image/png",
        "status": status,
        "attempts": attempts,
        "classification": classification,
        "response": response,
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True))

    deleted = False
    if status and 200 <= status < 300 and classification.get("wearing_mask") is True:
        if delete_masked and image_path.exists():
            image_path.unlink()
            deleted = True

    return {
        "image": str(image_path),
        "response_json": str(output_path),
        "status": status,
        "classification": classification,
        "deleted": deleted,
        "cached": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="Images or image directories to classify.")
    parser.add_argument("--manifest", help="CSV manifest from extract_faces_dlib.py.")
    parser.add_argument("--responses-dir", default="mask-detection/hk", help="Where to save Gemini JSON responses.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--name-prefix", help="Only process images whose filename starts with this prefix.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.5)
    parser.add_argument("--delete-masked", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment/.env")

    responses_dir = Path(args.responses_dir)
    responses_dir.mkdir(parents=True, exist_ok=True)

    images = [path for path in images_from_args(args) if path.exists()]
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit("No images to classify")

    results = []
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_image,
                image_path=image_path,
                responses_dir=responses_dir,
                api_key=api_key,
                model=args.model,
                timeout=args.timeout,
                delete_masked=args.delete_masked,
                force=args.force,
                retries=args.retries,
                retry_delay=args.retry_delay,
            ): image_path
            for image_path in images
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            classification = result["classification"]
            cached = " cached" if result.get("cached") else ""
            print(
                f"{completed}/{len(images)} {Path(result['image']).name}: "
                f"mask={classification.get('wearing_mask')} "
                f"confidence={classification.get('confidence')} "
                f"deleted={result['deleted']} status={result['status']}{cached}",
                flush=True,
            )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "delete_masked": args.delete_masked,
        "total": len(results),
        "masked": sum(1 for result in results if result["classification"].get("wearing_mask") is True),
        "unmasked": sum(1 for result in results if result["classification"].get("wearing_mask") is False),
        "unknown": sum(1 for result in results if result["classification"].get("wearing_mask") is None),
        "deleted": sum(1 for result in results if result["deleted"]),
        "results": results,
    }
    summary_path = responses_dir / f"summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
