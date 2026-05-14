#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate aligned color face images by applying saved outline alignment "
            "transforms directly to the original archive images."
        )
    )
    parser.add_argument("--alignments", default="landmarks/alignment")
    parser.add_argument("--manifest", action="append", default=["faces/_manifest.csv"])
    parser.add_argument("--output", default="faces-aligned")
    parser.add_argument("--format", choices=("png", "jpg"), default="png")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_manifest_rows(paths: list[str]) -> dict[str, dict[str, str]]:
    rows_by_output: dict[str, dict[str, str]] = {}
    for path_text in paths:
        path = Path(path_text)
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                output = row.get("output")
                if not output:
                    continue
                rows_by_output[Path(output).name] = row
    return rows_by_output


def float_field(row: dict[str, str], key: str) -> float:
    return float(row[key])


def int_field(row: dict[str, str], key: str) -> int:
    return int(round(float_field(row, key)))


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as loaded:
            return ImageOps.exif_transpose(loaded).size
    except (OSError, UnidentifiedImageError):
        return None


def source_bounds_for_output(
    output_size: tuple[int, int],
    crop_box: tuple[float, float, float, float],
    face_size: tuple[int, int],
    scale: float,
    dx: float,
    dy: float,
) -> dict[str, float]:
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    face_width, face_height = face_size
    output_width, output_height = output_size

    source_xs = [
        crop_left + ((x - dx) / scale) * crop_width / face_width
        for x in (0, output_width)
    ]
    source_ys = [
        crop_top + ((y - dy) / scale) * crop_height / face_height
        for y in (0, output_height)
    ]
    return {
        "left": min(source_xs),
        "top": min(source_ys),
        "right": max(source_xs),
        "bottom": max(source_ys),
    }


def edge_pad_image(
    image: Image.Image,
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
) -> Image.Image:
    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return image

    width, height = image.size
    padded = Image.new(
        image.mode,
        (width + pad_left + pad_right, height + pad_top + pad_bottom),
    )
    padded.paste(image, (pad_left, pad_top))

    if pad_left:
        padded.paste(
            image.crop((0, 0, 1, height)).resize((pad_left, height)),
            (0, pad_top),
        )
    if pad_right:
        padded.paste(
            image.crop((width - 1, 0, width, height)).resize((pad_right, height)),
            (pad_left + width, pad_top),
        )
    if pad_top:
        padded.paste(
            image.crop((0, 0, width, 1)).resize((width, pad_top)),
            (pad_left, 0),
        )
    if pad_bottom:
        padded.paste(
            image.crop((0, height - 1, width, height)).resize((width, pad_bottom)),
            (pad_left, pad_top + height),
        )

    if pad_left and pad_top:
        padded.paste(
            image.crop((0, 0, 1, 1)).resize((pad_left, pad_top), Image.Resampling.NEAREST),
            (0, 0),
        )
    if pad_right and pad_top:
        padded.paste(
            image.crop((width - 1, 0, width, 1)).resize(
                (pad_right, pad_top),
                Image.Resampling.NEAREST,
            ),
            (pad_left + width, 0),
        )
    if pad_left and pad_bottom:
        padded.paste(
            image.crop((0, height - 1, 1, height)).resize(
                (pad_left, pad_bottom),
                Image.Resampling.NEAREST,
            ),
            (0, pad_top + height),
        )
    if pad_right and pad_bottom:
        padded.paste(
            image.crop((width - 1, height - 1, width, height)).resize(
                (pad_right, pad_bottom),
                Image.Resampling.NEAREST,
            ),
            (pad_left + width, pad_top + height),
        )

    return padded


def render_aligned_from_source(
    source_path: Path,
    output_path: Path,
    crop_box: tuple[float, float, float, float],
    face_size: tuple[int, int],
    output_size: tuple[int, int],
    scale: float,
    dx: float,
    dy: float,
    image_format: str,
    quality: int,
) -> dict[str, int]:
    if scale <= 0:
        scale = 1.0

    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    face_width, face_height = face_size

    x_scale = crop_width / (face_width * scale)
    y_scale = crop_height / (face_height * scale)
    x_offset = crop_left - dx * crop_width / (face_width * scale)
    y_offset = crop_top - dy * crop_height / (face_height * scale)

    with Image.open(source_path) as loaded:
        source = ImageOps.exif_transpose(loaded).convert("RGB")

    bounds = source_bounds_for_output(
        output_size,
        crop_box,
        face_size,
        scale,
        dx,
        dy,
    )
    pad_left = math.ceil(-bounds["left"]) + 4 if bounds["left"] < 0 else 0
    pad_top = math.ceil(-bounds["top"]) + 4 if bounds["top"] < 0 else 0
    pad_right = math.ceil(bounds["right"] - source.width) + 4 if bounds["right"] > source.width else 0
    pad_bottom = (
        math.ceil(bounds["bottom"] - source.height) + 4
        if bounds["bottom"] > source.height
        else 0
    )
    source = edge_pad_image(source, pad_left, pad_top, pad_right, pad_bottom)

    affine = (
        x_scale,
        0,
        x_offset + pad_left,
        0,
        y_scale,
        y_offset + pad_top,
    )

    aligned = source.transform(
        output_size,
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpg":
        aligned.save(output_path, "JPEG", quality=quality, subsampling=1)
    else:
        aligned.save(output_path, "PNG")
    return {
        "left": pad_left,
        "top": pad_top,
        "right": pad_right,
        "bottom": pad_bottom,
    }


def main() -> int:
    args = parse_args()
    alignments_dir = Path(args.alignments)
    output_dir = Path(args.output)
    rows_by_output = load_manifest_rows(args.manifest)
    if not rows_by_output:
        print("No manifest rows loaded.", flush=True)
        return 1

    alignment_paths = sorted(alignments_dir.glob("*.json"))
    if not alignment_paths:
        print(f"No alignment JSON found in {alignments_dir}", flush=True)
        return 1

    rendered = 0
    skipped_existing = 0
    skipped_missing_manifest = 0
    skipped_missing_source = 0
    skipped_missing_size = 0
    skipped_bad_json = 0
    out_of_source_bounds = 0
    records: list[dict[str, Any]] = []
    extension = ".jpg" if args.format == "jpg" else ".png"

    for index, alignment_path in enumerate(alignment_paths, start=1):
        try:
            alignment = json.loads(alignment_path.read_text())
        except json.JSONDecodeError:
            skipped_bad_json += 1
            continue

        pair_id = alignment.get("pair_id")
        face_path = Path(str(alignment.get("face_path", "")))
        outline_path = Path(str(alignment.get("outline_path", "")))
        transform = alignment.get("transform", {})
        if not isinstance(pair_id, str) or not isinstance(transform, dict):
            skipped_bad_json += 1
            continue

        output_path = output_dir / f"{pair_id}{extension}"
        if output_path.exists() and not args.force:
            skipped_existing += 1
            continue

        manifest_row = rows_by_output.get(face_path.name)
        if manifest_row is None:
            skipped_missing_manifest += 1
            continue

        source_path = Path(manifest_row["source"])
        if not source_path.is_file():
            skipped_missing_source += 1
            continue

        face_size = image_size(face_path)
        outline_size = image_size(outline_path)
        if face_size is None or outline_size is None:
            skipped_missing_size += 1
            continue

        crop_box = (
            float_field(manifest_row, "crop_left"),
            float_field(manifest_row, "crop_top"),
            float_field(manifest_row, "crop_right"),
            float_field(manifest_row, "crop_bottom"),
        )
        scale = float(transform.get("scale", 1.0))
        dx = float(transform.get("dx", 0.0))
        dy = float(transform.get("dy", 0.0))

        padding = render_aligned_from_source(
            source_path,
            output_path,
            crop_box,
            face_size,
            outline_size,
            scale,
            dx,
            dy,
            args.format,
            args.quality,
        )

        source_size = (
            int_field(manifest_row, "source_width"),
            int_field(manifest_row, "source_height"),
        )
        bounds = source_bounds_for_output(
            outline_size,
            crop_box,
            face_size,
            scale,
            dx,
            dy,
        )
        outside = (
            bounds["left"] < 0
            or bounds["top"] < 0
            or bounds["right"] > source_size[0]
            or bounds["bottom"] > source_size[1]
        )
        if outside:
            out_of_source_bounds += 1

        rendered += 1
        records.append(
            {
                "pair_id": pair_id,
                "source": str(source_path),
                "face": str(face_path),
                "outline": str(outline_path),
                "output": str(output_path),
                "crop_box": list(crop_box),
                "face_size": list(face_size),
                "output_size": list(outline_size),
                "transform": {"scale": scale, "dx": dx, "dy": dy},
                "source_bounds_sampled": bounds,
                "source_size": list(source_size),
                "out_of_source_bounds": outside,
                "edge_padding_applied": padding,
            }
        )
        print(f"{rendered:04d}/{len(alignment_paths)} {output_path}", flush=True)

    manifest_path = output_dir / "_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "alignments": str(alignments_dir),
                "manifests": args.manifest,
                "output": str(output_dir),
                "format": args.format,
                "total_alignment_files": len(alignment_paths),
                "rendered": rendered,
                "skipped_existing": skipped_existing,
                "skipped_missing_manifest": skipped_missing_manifest,
                "skipped_missing_source": skipped_missing_source,
                "skipped_missing_size": skipped_missing_size,
                "skipped_bad_json": skipped_bad_json,
                "out_of_source_bounds": out_of_source_bounds,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "summary "
        f"alignments={len(alignment_paths)} rendered={rendered} "
        f"skipped_existing={skipped_existing} "
        f"skipped_missing_manifest={skipped_missing_manifest} "
        f"skipped_missing_source={skipped_missing_source} "
        f"skipped_missing_size={skipped_missing_size} "
        f"skipped_bad_json={skipped_bad_json} "
        f"out_of_source_bounds={out_of_source_bounds} "
        f"manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
