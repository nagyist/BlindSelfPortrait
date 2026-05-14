#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import os
import random
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}

PROMPT = (
    "Convert this image into clean black and white coloring book line art. "
    "Align the output as closely as possible to the input image, preserving the "
    "subject placement, scale, pose, crop, facial features, and composition. Use "
    "bold clear outlines, simple interior contour lines, white background, no "
    "color, no shading, no gray fill, printable children's coloring page style. "
    "Do not copy or reproduce any text, letters, numbers, captions, labels, "
    "signage, watermarks, brand marks, or logos from the input image; omit them "
    "or replace them with blank line-art shapes. "
    "Render the hair as an outline, do not render internal texture on hair, "
    "eyebrows, or clothes."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick one random face from faces/, submit it to gpt-image-2, and save "
            "the image-edit response JSON plus decoded image in outline/."
        )
    )
    parser.add_argument("--input", default="faces", help="Directory of face images.")
    parser.add_argument(
        "--image",
        default=None,
        help="Specific image path, filename, or stem to use instead of a random face.",
    )
    parser.add_argument("--output", default="outline", help="Directory for raw responses.")
    parser.add_argument("--env-file", default=".env", help="File containing OPENAI_API_KEY.")
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument(
        "--quality",
        default="low",
        choices=("low",),
        help="Rendering quality. This script always uses low.",
    )
    parser.add_argument("--output-format", default="png")
    parser.add_argument("--background", default="opaque")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected image and request fields without calling the API.",
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


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_image(paths: list[Path], seed: int | None) -> Path:
    if not paths:
        raise ValueError("No supported images found.")
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    return rng.choice(paths)


def resolve_image(input_dir: Path, image_arg: str | None, seed: int | None) -> Path:
    images = iter_images(input_dir)
    if image_arg is None:
        return choose_image(images, seed)

    requested = Path(image_arg).expanduser()
    if requested.is_file():
        return requested

    candidate = input_dir / image_arg
    if candidate.is_file():
        return candidate

    matches = [
        path
        for path in images
        if path.name == image_arg or path.stem == image_arg or str(path) == image_arg
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Image reference matched multiple files: {image_arg}")

    raise ValueError(f"Image not found: {image_arg}")


def quote_header_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def multipart_body(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"----BlindSelfPortrait{secrets.token_hex(16)}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{quote_header_value(name)}"\r\n\r\n'.encode(
                "utf-8"
            )
        )
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{quote_header_value(file_field)}"; '
            f'filename="{quote_header_value(file_path.name)}"\r\n'
        ).encode("utf-8")
    )
    chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def filename_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in value)


def unique_response_path(
    output_dir: Path,
    source_image: Path,
    model: str,
    suffix: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_token = filename_token(model)
    base = f"{source_image.stem}__{model_token}__{timestamp}{suffix}.json"
    candidate = output_dir / base
    counter = 1
    while candidate.exists():
        candidate = output_dir / (
            f"{source_image.stem}__{model_token}__{timestamp}{suffix}-{counter}.json"
        )
        counter += 1
    return candidate


def save_response(
    output_dir: Path,
    source_image: Path,
    model: str,
    suffix: str,
    body: bytes,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = unique_response_path(output_dir, source_image, model, suffix)
    response_path.write_bytes(body)
    return response_path


def response_payload(body: bytes) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Response was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Response JSON was not an object.")
    return payload


def decode_response_image(payload: dict[str, object]) -> bytes:
    try:
        image_base64 = payload["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Response did not contain data[0].b64_json.") from exc

    if not isinstance(image_base64, str) or not image_base64:
        raise ValueError("Response image data was empty.")

    try:
        return base64.b64decode(image_base64, validate=True)
    except binascii.Error as exc:
        raise ValueError("Response image data was not valid base64.") from exc


def save_sanitized_response(
    output_dir: Path,
    source_image: Path,
    model: str,
    payload: dict[str, object],
    request_metadata: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = unique_response_path(output_dir, source_image, model, "")
    sanitized = dict(payload)
    sanitized.pop("data", None)
    sanitized["request"] = request_metadata
    response_path.write_text(json.dumps(sanitized, indent=2) + "\n")
    return response_path


def save_response_image(response_path: Path, output_format: str, image_bytes: bytes) -> Path:
    image_extension = filename_token(output_format.lower()) or "png"
    image_path = response_path.with_suffix(f".{image_extension}")
    image_path.write_bytes(image_bytes)
    return image_path


def post_image_edit(
    api_key: str,
    image_path: Path,
    model: str,
    size: str,
    quality: str,
    output_format: str,
    background: str,
    timeout: int,
) -> tuple[int, bytes, str | None]:
    body, content_type = multipart_body(
        {
            "model": model,
            "prompt": PROMPT,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "background": background,
        },
        "image[]",
        image_path,
    )
    request = Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read(), response.headers.get("x-request-id")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    env_file = Path(args.env_file).expanduser()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    try:
        image_path = resolve_image(input_dir, args.image, args.seed)
    except ValueError as exc:
        print(f"{exc} Directory: {input_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"selected_image={image_path}")
        print(f"model={args.model}")
        print(f"size={args.size}")
        print(f"quality={args.quality}")
        print(f"output_format={args.output_format}")
        print(f"background={args.background}")
        print(f"output_dir={output_dir}")
        return 0

    load_env_file(env_file)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"OPENAI_API_KEY not found in environment or {env_file}", file=sys.stderr)
        return 1

    print(f"selected_image={image_path}")
    request_started_at = datetime.now().astimezone()
    request_started = time.perf_counter()
    try:
        status, body, request_id = post_image_edit(
            api_key=api_key,
            image_path=image_path,
            model=args.model,
            size=args.size,
            quality=args.quality,
            output_format=args.output_format,
            background=args.background,
            timeout=args.timeout,
        )
        request_completed_at = datetime.now().astimezone()
        request_duration_seconds = time.perf_counter() - request_started
    except HTTPError as exc:
        request_completed_at = datetime.now().astimezone()
        request_duration_seconds = time.perf_counter() - request_started
        body = exc.read()
        response_path = save_response(
            output_dir,
            image_path,
            args.model,
            "__error",
            body,
        )
        request_id = exc.headers.get("x-request-id")
        print(f"HTTP {exc.code}; raw response saved to {response_path}", file=sys.stderr)
        print(f"request_duration_seconds={request_duration_seconds:.3f}", file=sys.stderr)
        if request_id:
            print(f"x-request-id={request_id}", file=sys.stderr)
        return 1
    except URLError as exc:
        request_completed_at = datetime.now().astimezone()
        request_duration_seconds = time.perf_counter() - request_started
        print(f"Request failed: {exc}", file=sys.stderr)
        print(f"request_duration_seconds={request_duration_seconds:.3f}", file=sys.stderr)
        return 1

    try:
        payload = response_payload(body)
        image_bytes = decode_response_image(payload)
    except ValueError as exc:
        print(f"Could not save decoded image: {exc}", file=sys.stderr)
        return 1

    request_metadata = {
        "source_image": str(image_path),
        "model": args.model,
        "size": args.size,
        "quality": args.quality,
        "output_format": args.output_format,
        "background": args.background,
        "prompt": PROMPT,
        "status": status,
        "x_request_id": request_id,
        "started_at": request_started_at.isoformat(timespec="seconds"),
        "completed_at": request_completed_at.isoformat(timespec="seconds"),
        "duration_seconds": round(request_duration_seconds, 3),
    }
    response_path = save_sanitized_response(
        output_dir,
        image_path,
        args.model,
        payload,
        request_metadata,
    )
    png_path = save_response_image(response_path, args.output_format, image_bytes)

    print(f"status={status}")
    if request_id:
        print(f"x-request-id={request_id}")
    print(f"request_duration_seconds={request_duration_seconds:.3f}")
    print(f"response_json={response_path}")
    print(f"image={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
