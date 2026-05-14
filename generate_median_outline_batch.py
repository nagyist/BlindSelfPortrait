#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path


DEFAULT_BLUR_FIELD = "face_blur_64_laplacian_variance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate outlines for faces closest to median manifest blurriness."
    )
    parser.add_argument("--manifest", default="faces/_manifest.csv")
    parser.add_argument("--input", default="faces")
    parser.add_argument("--output", default="outline")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--blur-field", default=DEFAULT_BLUR_FIELD)
    parser.add_argument("--script", default="generate_outline_response.py")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--resume-log",
        default=None,
        help="Existing JSONL progress log whose successful image rows should be skipped.",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path, input_dir: Path, blur_field: str) -> list[tuple[float, Path]]:
    rows: list[tuple[float, Path]] = []
    with manifest_path.open(newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        for row in reader:
            try:
                blur = float(row[blur_field])
            except (KeyError, TypeError, ValueError):
                continue

            path = Path(row.get("output", ""))
            if not path.is_file():
                path = input_dir / path.name
            if path.is_file():
                rows.append((blur, path))
    return rows


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("Cannot compute median of an empty list.")
    if count % 2:
        return ordered[count // 2]
    return (ordered[count // 2 - 1] + ordered[count // 2]) / 2


def selected_faces(rows: list[tuple[float, Path]], limit: int) -> tuple[float, list[tuple[float, Path]]]:
    median_blur = median([blur for blur, _ in rows])
    selected = sorted(rows, key=lambda item: (abs(item[0] - median_blur), item[1].name))[
        :limit
    ]
    return median_blur, selected


def write_selection_csv(
    path: Path,
    median_blur: float,
    selected: list[tuple[float, Path]],
) -> None:
    with path.open("w", newline="") as selection_file:
        writer = csv.DictWriter(
            selection_file,
            fieldnames=["rank", "image", "blur", "median_blur", "delta"],
        )
        writer.writeheader()
        for rank, (blur, image_path) in enumerate(selected, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "image": str(image_path),
                    "blur": f"{blur:.6f}",
                    "median_blur": f"{median_blur:.6f}",
                    "delta": f"{abs(blur - median_blur):.6f}",
                }
            )


def parse_generator_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            parsed[key.strip()] = value.strip()
    return parsed


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as log_file:
        log_file.write(json.dumps(row, sort_keys=True) + "\n")


def completed_images_from_log(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()

    completed: set[str] = set()
    with path.open() as log_file:
        for line in log_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("returncode") == 0 and row.get("image"):
                completed.add(str(row["image"]))
    return completed


def run_generator(
    rank: int,
    total: int,
    blur: float,
    median_blur: float,
    image_path: Path,
    script_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--image",
            str(image_path),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = time.perf_counter() - started
    parsed = parse_generator_output(result.stdout + "\n" + result.stderr)
    return {
        "rank": rank,
        "total": total,
        "image": str(image_path),
        "blur": round(blur, 6),
        "median_blur": round(median_blur, 6),
        "delta": round(abs(blur - median_blur), 6),
        "returncode": result.returncode,
        "duration_seconds": round(duration, 3),
        "response_json": parsed.get("response_json"),
        "image_output": parsed.get("image"),
        "request_duration_seconds": parsed.get("request_duration_seconds"),
        "x_request_id": parsed.get("x-request-id"),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    script_path = Path(args.script)
    concurrency = max(1, args.concurrency)

    rows = load_manifest(manifest_path, input_dir, args.blur_field)
    if not rows:
        print(f"No manifest rows with existing face files: {manifest_path}", file=sys.stderr)
        return 1

    median_blur, selected = selected_faces(rows, args.limit)
    resume_log = Path(args.resume_log) if args.resume_log else None
    completed_images = completed_images_from_log(resume_log)
    pending = [
        (rank, blur, image_path)
        for rank, (blur, image_path) in enumerate(selected, 1)
        if str(image_path) not in completed_images
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    selection_path = output_dir / f"median_closest_{len(selected)}__{batch_id}.csv"
    log_path = output_dir / f"median_closest_{len(selected)}__{batch_id}.jsonl"
    write_selection_csv(selection_path, median_blur, selected)

    print(f"manifest_rows={len(rows)}", flush=True)
    print(f"median_blur={median_blur:.6f}", flush=True)
    print(f"selected_count={len(selected)}", flush=True)
    print(f"already_completed={len(completed_images)}", flush=True)
    print(f"pending_count={len(pending)}", flush=True)
    print(f"concurrency={concurrency}", flush=True)
    print(f"selection_csv={selection_path}", flush=True)
    print(f"progress_jsonl={log_path}", flush=True)
    if resume_log is not None:
        print(f"resume_log={resume_log}", flush=True)

    if args.dry_run:
        for rank, blur, image_path in pending[:20]:
            print(
                f"DRY {rank:03d}/{len(selected)} blur={blur:.6f} "
                f"delta={abs(blur - median_blur):.6f} image={image_path}",
                flush=True,
            )
        return 0

    consecutive_failures = 0
    completed = 0
    failed = 0
    batch_started = time.perf_counter()
    next_index = 0
    futures = {}
    stop_submitting = False

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal next_index
        if next_index >= len(pending):
            return False
        rank, blur, image_path = pending[next_index]
        next_index += 1
        print(
            f"START {rank:03d}/{len(selected)} blur={blur:.6f} "
            f"delta={abs(blur - median_blur):.6f} image={image_path}",
            flush=True,
        )
        future = executor.submit(
            run_generator,
            rank,
            len(selected),
            blur,
            median_blur,
            image_path,
            script_path,
            input_dir,
            output_dir,
        )
        futures[future] = rank
        return True

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(min(concurrency, len(pending))):
            submit_next(executor)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                row = future.result()
                append_jsonl(log_path, row)
                rank = int(row["rank"])
                duration = float(row["duration_seconds"])
                if row["returncode"] == 0:
                    completed += 1
                    consecutive_failures = 0
                    print(
                        f"DONE {rank:03d}/{len(selected)} duration={duration:.3f}s "
                        f"image={row.get('image_output')}",
                        flush=True,
                    )
                else:
                    failed += 1
                    consecutive_failures += 1
                    print(
                        f"FAIL {rank:03d}/{len(selected)} returncode={row['returncode']} "
                        f"duration={duration:.3f}s",
                        flush=True,
                    )
                    if row.get("stderr"):
                        print(str(row["stderr"]), file=sys.stderr, flush=True)
                    if consecutive_failures >= args.max_consecutive_failures:
                        print(
                            f"Stopping after {consecutive_failures} consecutive failures.",
                            file=sys.stderr,
                            flush=True,
                        )
                        stop_submitting = True

                if not stop_submitting:
                    submit_next(executor)

            if stop_submitting:
                for future in futures:
                    future.cancel()
                break

    elapsed = time.perf_counter() - batch_started
    print(
        f"summary completed={completed} failed={failed} "
        f"elapsed_seconds={elapsed:.3f} progress_jsonl={log_path}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
