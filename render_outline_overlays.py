#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multiply-blend generated outline PNGs over their source face crops."
    )
    parser.add_argument("--outline", default="outline", help="Directory containing outline PNGs.")
    parser.add_argument("--faces", default="faces", help="Fallback directory for source faces.")
    parser.add_argument("--output", default="overlaid", help="Directory for overlay renders.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--prefer-faces-dir",
        action="store_true",
        help="Prefer matching files from --faces before source_image in outline JSON.",
    )
    return parser.parse_args()


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


def source_from_faces_dir(outline_path: Path, faces_dir: Path) -> Path | None:
    exact = faces_dir / outline_path.name
    if exact.is_file():
        return exact

    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = faces_dir / f"{outline_path.stem}{suffix}"
        if candidate.is_file():
            return candidate

    marker = "__gpt-image-2__"
    if marker not in outline_path.stem:
        return None
    face_stem = outline_path.stem.split(marker, 1)[0]
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = faces_dir / f"{face_stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def source_face(outline_path: Path, faces_dir: Path, prefer_faces_dir: bool) -> Path | None:
    if prefer_faces_dir:
        source = source_from_faces_dir(outline_path, faces_dir)
        if source is not None:
            return source

    source = source_from_json(outline_path)
    if source is not None and source.is_file():
        return source
    return source_from_faces_dir(outline_path, faces_dir)


def multiply_overlay(face_path: Path, outline_path: Path, output_path: Path) -> None:
    with Image.open(face_path) as face_loaded, Image.open(outline_path) as outline_loaded:
        face = ImageOps.exif_transpose(face_loaded).convert("RGB")
        outline = ImageOps.exif_transpose(outline_loaded).convert("RGB")

    if outline.size != face.size:
        outline = outline.resize(face.size, Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ImageChops.multiply(face, outline).save(output_path, "PNG")


def main() -> int:
    args = parse_args()
    outline_dir = Path(args.outline)
    faces_dir = Path(args.faces)
    output_dir = Path(args.output)

    if not outline_dir.is_dir():
        print(f"Outline directory not found: {outline_dir}", file=sys.stderr)
        return 1
    if not faces_dir.is_dir():
        print(f"Faces directory not found: {faces_dir}", file=sys.stderr)
        return 1

    rendered = 0
    skipped_existing = 0
    skipped_missing_source = 0
    skipped_unreadable = 0

    for outline_path in outline_pngs(outline_dir):
        if args.limit is not None and rendered >= args.limit:
            break

        output_path = output_dir / outline_path.name
        if args.skip_existing and output_path.exists():
            skipped_existing += 1
            continue

        face_path = source_face(outline_path, faces_dir, args.prefer_faces_dir)
        if face_path is None:
            skipped_missing_source += 1
            continue

        try:
            multiply_overlay(face_path, outline_path, output_path)
        except (OSError, UnidentifiedImageError):
            skipped_unreadable += 1
            continue

        rendered += 1
        print(f"{rendered:04d} {output_path} source={face_path}")

    print(
        "summary "
        f"rendered={rendered} skipped_existing={skipped_existing} "
        f"skipped_missing_source={skipped_missing_source} "
        f"skipped_unreadable={skipped_unreadable} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
