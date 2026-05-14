#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import dlib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class LandmarkFace:
    face: FaceBox
    points: tuple[tuple[float, float], ...]
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    eye_center: tuple[float, float]
    nose: tuple[float, float]
    interocular_distance: float

    @property
    def anchor(self) -> tuple[float, float]:
        return (
            (self.eye_center[0] + self.nose[0]) / 2,
            (self.eye_center[1] + self.nose[1]) / 2,
        )

    @property
    def point_center(self) -> tuple[float, float]:
        return (
            sum(point[0] for point in self.points) / len(self.points),
            sum(point[1] for point in self.points) / len(self.points),
        )

    @property
    def average_eye_nose_distance(self) -> float:
        left = np.hypot(self.left_eye[0] - self.nose[0], self.left_eye[1] - self.nose[1])
        right = np.hypot(self.right_eye[0] - self.nose[0], self.right_eye[1] - self.nose[1])
        return float((left + right) / 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 1024x1024 head-and-shoulders crops using dlib landmarks."
    )
    parser.add_argument("--input", default="archive", help="Image directory or Finder alias.")
    parser.add_argument("--output", default="faces", help="Directory for cropped face images.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix to prepend to each output filename.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many saved crops.")
    parser.add_argument(
        "--only",
        default=None,
        help="Only process images whose archive-relative path or output filename contains this text.",
    )
    parser.add_argument(
        "--only-list",
        default=None,
        help="Text file of source-relative paths, output filenames, stems, or substrings to process.",
    )
    parser.add_argument("--manifest", default="_manifest.csv", help="Manifest filename inside output.")
    parser.add_argument(
        "--shape-predictor",
        default="models/shape_predictor_5_face_landmarks.dat",
        help="Path to dlib's 5-point landmark predictor .dat file.",
    )
    parser.add_argument(
        "--landmark-scale",
        type=float,
        default=6.18,
        help="Crop side length as a multiple of the average eye-to-nose landmark distance.",
    )
    parser.add_argument("--max-detect-dim", type=int, default=1600)
    parser.add_argument("--upsample", type=int, default=1)
    parser.add_argument(
        "--face-center-region-scale",
        type=float,
        default=1.0,
        help="Only keep detected faces whose bbox center is inside a centered square of this scale.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def resolve_finder_alias(path: Path) -> Path:
    if path.is_dir():
        return path
    if sys.platform != "darwin" or not path.exists():
        return path

    script = (
        'tell application "Finder" to get POSIX path of '
        f'(original item of alias POSIX file "{escape_applescript_string(str(path))}" as alias)'
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return path


def iter_images(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_for_detection(image: Image.Image, max_detect_dim: int) -> tuple[Image.Image, float]:
    max_dim = max(image.size)
    if max_dim <= max_detect_dim:
        return image, 1.0

    scale = max_detect_dim / max_dim
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BILINEAR), scale


def detect_faces(
    detector: dlib.fhog_object_detector,
    image: Image.Image,
    max_detect_dim: int,
    upsample: int,
) -> list[FaceBox]:
    detect_image, scale = image_for_detection(image, max_detect_dim)
    rects = detector(np.asarray(detect_image), upsample)

    faces: list[FaceBox] = []
    for rect in rects:
        faces.append(
            FaceBox(
                left=round(rect.left() / scale),
                top=round(rect.top() / scale),
                right=round((rect.right() + 1) / scale),
                bottom=round((rect.bottom() + 1) / scale),
            )
        )
    return faces


def most_central_face(faces: list[FaceBox], width: int, height: int) -> FaceBox:
    image_cx = width / 2
    image_cy = height / 2

    def score(face: FaceBox) -> tuple[float, int]:
        face_cx, face_cy = face.center
        dx = (face_cx - image_cx) / width
        dy = (face_cy - image_cy) / height
        return (dx * dx + dy * dy, -face.area)

    return min(faces, key=score)


def most_central_landmark_face(
    landmark_faces: list[LandmarkFace], width: int, height: int
) -> LandmarkFace:
    image_cx = width / 2
    image_cy = height / 2

    def score(landmark_face: LandmarkFace) -> tuple[float, int]:
        face_cx, face_cy = landmark_face.point_center
        dx = (face_cx - image_cx) / width
        dy = (face_cy - image_cy) / height
        return (dx * dx + dy * dy, -landmark_face.face.area)

    return min(landmark_faces, key=score)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def predict_landmarks(
    predictor: dlib.shape_predictor,
    image_array: np.ndarray,
    face: FaceBox,
    width: int,
    height: int,
) -> LandmarkFace:
    rectangle = dlib.rectangle(
        clamp(face.left, 0, width - 1),
        clamp(face.top, 0, height - 1),
        clamp(face.right - 1, 0, width - 1),
        clamp(face.bottom - 1, 0, height - 1),
    )
    shape = predictor(image_array, rectangle)
    points = tuple((float(shape.part(i).x), float(shape.part(i).y)) for i in range(5))
    eye_points = sorted(points[:4], key=lambda point: point[0])
    left_eye = (
        (eye_points[0][0] + eye_points[1][0]) / 2,
        (eye_points[0][1] + eye_points[1][1]) / 2,
    )
    right_eye = (
        (eye_points[2][0] + eye_points[3][0]) / 2,
        (eye_points[2][1] + eye_points[3][1]) / 2,
    )
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2,
    )
    nose = points[4]
    interocular_distance = float(
        np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    )
    return LandmarkFace(
        face=face,
        points=points,
        left_eye=left_eye,
        right_eye=right_eye,
        eye_center=eye_center,
        nose=nose,
        interocular_distance=interocular_distance,
    )


def head_and_shoulders_crop(
    landmark_face: LandmarkFace,
    width: int,
    height: int,
    landmark_scale: float,
) -> tuple[int, int, int, int]:
    side = round(landmark_face.average_eye_nose_distance * landmark_scale)
    side = max(side, round(landmark_face.average_eye_nose_distance * 2.5))

    center_x, center_y = landmark_face.point_center
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)

    return (left, top, left + side, top + side)


def crop_in_bounds(
    crop_box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    return crop_left >= 0 and crop_top >= 0 and crop_right <= width and crop_bottom <= height


def blur_laplacian_variance(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


def source_box_to_image_box(
    source_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    source_left, source_top, source_right, source_bottom = source_box
    image_width, image_height = image_size
    scale_x = image_width / (crop_right - crop_left)
    scale_y = image_height / (crop_bottom - crop_top)
    left = round((source_left - crop_left) * scale_x)
    top = round((source_top - crop_top) * scale_y)
    right = round((source_right - crop_left) * scale_x)
    bottom = round((source_bottom - crop_top) * scale_y)
    return (
        clamp(left, 0, image_width),
        clamp(top, 0, image_height),
        clamp(right, 0, image_width),
        clamp(bottom, 0, image_height),
    )


def face_blur_64_laplacian_variance(
    cropped: Image.Image,
    face: FaceBox,
    crop_box: tuple[int, int, int, int],
) -> float:
    face_box = source_box_to_image_box(
        (face.left, face.top, face.right, face.bottom),
        crop_box,
        cropped.size,
    )
    left, top, right, bottom = face_box
    if right - left < 3 or bottom - top < 3:
        return 0.0
    face_crop = cropped.crop(face_box).resize((64, 64), Image.Resampling.LANCZOS)
    return blur_laplacian_variance(face_crop)


def output_name(root: Path, source: Path, prefix: str = "") -> str:
    relative = source.relative_to(root if root.is_dir() else root.parent)
    stem_parts = list(relative.with_suffix("").parts)
    return prefix + "__".join(stem_parts) + ".jpg"


def matches_only_filter(root: Path, source: Path, only: str | None) -> bool:
    if only is None:
        return True
    output_filename = output_name(root, source)
    try:
        relative = str(source.relative_to(root if root.is_dir() else root.parent))
    except ValueError:
        relative = source.name
    normalized_only = only.replace("\\", "/")
    normalized_relative = relative.replace("\\", "/")
    return normalized_only in normalized_relative or normalized_only in output_filename


def load_only_list(path: str | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip().replace("\\", "/")
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def matches_only_list(root: Path, source: Path, output_filename: str, only_list: list[str]) -> bool:
    if not only_list:
        return True
    try:
        relative = str(source.relative_to(root if root.is_dir() else root.parent))
    except ValueError:
        relative = source.name
    candidates = {
        relative.replace("\\", "/"),
        source.name,
        source.stem,
        output_filename,
        Path(output_filename).stem,
    }
    return any(
        token in candidate
        for token in only_list
        for candidate in candidates
    )


def centered_square_region(
    width: int,
    height: int,
    scale: float,
) -> tuple[float, float, float, float]:
    side = min(width, height) * scale
    center_x = width / 2
    center_y = height / 2
    return (
        center_x - side / 2,
        center_y - side / 2,
        center_x + side / 2,
        center_y + side / 2,
    )


def face_center_in_region(face: FaceBox, region: tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = region
    center_x, center_y = face.center
    return left <= center_x <= right and top <= center_y <= bottom


def main() -> int:
    args = parse_args()
    input_root = resolve_finder_alias(Path(args.input).expanduser()).resolve()
    output_root = Path(args.output).expanduser()
    shape_predictor_path = Path(args.shape_predictor).expanduser()

    if not input_root.is_dir() and not input_root.is_file():
        print(f"Input is not a directory or image file: {input_root}", file=sys.stderr)
        return 1
    if not shape_predictor_path.is_file():
        print(f"Shape predictor not found: {shape_predictor_path}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest
    only_list = load_only_list(args.only_list)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(shape_predictor_path))

    saved = 0
    scanned = 0
    skipped_no_face = 0
    skipped_no_landmarks = 0
    skipped_out_of_bounds = 0
    skipped_existing = 0
    skipped_unreadable = 0
    skipped_off_center = 0

    rows: list[dict[str, object]] = []
    for source in iter_images(input_root):
        if not matches_only_filter(input_root, source, args.only):
            continue
        if args.limit is not None and saved >= args.limit:
            break

        destination = output_root / output_name(input_root, source, args.output_prefix)
        if not matches_only_list(input_root, source, destination.name, only_list):
            continue
        scanned += 1
        if args.skip_existing and destination.exists():
            skipped_existing += 1
            continue

        try:
            with Image.open(source) as loaded:
                image = ImageOps.exif_transpose(loaded).convert("RGB")
        except (OSError, UnidentifiedImageError):
            skipped_unreadable += 1
            continue

        faces = detect_faces(detector, image, args.max_detect_dim, args.upsample)
        if args.face_center_region_scale < 1.0:
            center_region = centered_square_region(
                image.width,
                image.height,
                args.face_center_region_scale,
            )
            faces = [
                face
                for face in faces
                if face_center_in_region(face, center_region)
            ]
        else:
            center_region = (0.0, 0.0, float(image.width), float(image.height))
        if not faces:
            if args.face_center_region_scale < 1.0:
                skipped_off_center += 1
            else:
                skipped_no_face += 1
            continue

        image_array = np.asarray(image)
        landmark_faces = [
            predict_landmarks(predictor, image_array, face, image.width, image.height)
            for face in faces
        ]
        landmark_faces = [
            landmark_face
            for landmark_face in landmark_faces
            if landmark_face.average_eye_nose_distance > 0
        ]
        if not landmark_faces:
            skipped_no_landmarks += 1
            continue

        landmark_face = most_central_landmark_face(landmark_faces, image.width, image.height)
        face = landmark_face.face
        crop_box = head_and_shoulders_crop(
            landmark_face,
            image.width,
            image.height,
            args.landmark_scale,
        )
        if not crop_in_bounds(crop_box, image.width, image.height):
            skipped_out_of_bounds += 1
            continue

        cropped = image.crop(crop_box).resize((1024, 1024), Image.Resampling.LANCZOS)
        blur_score = face_blur_64_laplacian_variance(cropped, face, crop_box)
        cropped.save(destination, "JPEG", quality=95, subsampling=1)
        saved += 1

        rows.append(
            {
                "output": str(destination),
                "source": str(source),
                "faces_detected": len(faces),
                "selected_face_left": face.left,
                "selected_face_top": face.top,
                "selected_face_right": face.right,
                "selected_face_bottom": face.bottom,
                "left_eye_x": round(landmark_face.left_eye[0], 3),
                "left_eye_y": round(landmark_face.left_eye[1], 3),
                "right_eye_x": round(landmark_face.right_eye[0], 3),
                "right_eye_y": round(landmark_face.right_eye[1], 3),
                "eye_center_x": round(landmark_face.eye_center[0], 3),
                "eye_center_y": round(landmark_face.eye_center[1], 3),
                "nose_x": round(landmark_face.nose[0], 3),
                "nose_y": round(landmark_face.nose[1], 3),
                "point_center_x": round(landmark_face.point_center[0], 3),
                "point_center_y": round(landmark_face.point_center[1], 3),
                "interocular_distance": round(landmark_face.interocular_distance, 3),
                "average_eye_nose_distance": round(
                    landmark_face.average_eye_nose_distance, 3
                ),
                "crop_left": crop_box[0],
                "crop_top": crop_box[1],
                "crop_right": crop_box[2],
                "crop_bottom": crop_box[3],
                "source_width": image.width,
                "source_height": image.height,
                "face_blur_64_laplacian_variance": round(blur_score, 6),
                "face_center_region_scale": args.face_center_region_scale,
                "face_center_region_left": round(center_region[0], 3),
                "face_center_region_top": round(center_region[1], 3),
                "face_center_region_right": round(center_region[2], 3),
                "face_center_region_bottom": round(center_region[3], 3),
            }
        )
        print(f"{saved:04d} {destination}")

    fieldnames = [
        "output",
        "source",
        "faces_detected",
        "selected_face_left",
        "selected_face_top",
        "selected_face_right",
        "selected_face_bottom",
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
        "eye_center_x",
        "eye_center_y",
        "nose_x",
        "nose_y",
        "point_center_x",
        "point_center_y",
        "interocular_distance",
        "average_eye_nose_distance",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "source_width",
        "source_height",
        "face_blur_64_laplacian_variance",
        "face_center_region_scale",
        "face_center_region_left",
        "face_center_region_top",
        "face_center_region_right",
        "face_center_region_bottom",
    ]
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "summary "
        f"input={input_root} scanned={scanned} saved={saved} "
        f"skipped_no_face={skipped_no_face} skipped_existing={skipped_existing} "
        f"skipped_no_landmarks={skipped_no_landmarks} "
        f"skipped_out_of_bounds={skipped_out_of_bounds} "
        f"skipped_off_center={skipped_off_center} "
        f"skipped_unreadable={skipped_unreadable} "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
