#!/usr/bin/env python3
"""Convert black line art into routed centerline vector strokes.

The pipeline is intentionally dependency-light: it uses Pillow for image IO and
NumPy for image processing. It does not call Potrace, OpenCV, scikit-image, or
any of the older repo experiments.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)


@dataclass
class Stroke:
    points: np.ndarray
    widths: np.ndarray
    length: float
    median_width: float

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]


@dataclass
class Route:
    order: list[int]
    reversed_flags: list[bool]
    connector_length: float


@dataclass
class ConnectorModel:
    endpoint_positions: np.ndarray
    endpoint_widths: np.ndarray
    cost_matrix: np.ndarray
    straight_cost_matrix: np.ndarray
    straight_length_matrix: np.ndarray
    graph_length_matrix: np.ndarray
    snap_matrix: np.ndarray
    graph_points: np.ndarray
    graph_widths: np.ndarray
    graph: list[list[tuple[int, float]]]
    endpoint_nodes: list[int]
    retrace_weight: float
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_start_widths: np.ndarray
    segment_end_widths: np.ndarray
    raster_cost: np.ndarray | None = None
    raster_line: np.ndarray | None = None
    raster_step: int = 4
    raster_max_ratio: float = 2.75
    raster_min_line_fraction: float = 0.25
    raster_projection_distance: float = 7.0
    raster_min_projection_fraction: float = 0.8
    raster_path_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, bool]] = field(
        default_factory=dict
    )

    def cost(self, from_endpoint: int, to_endpoint: int) -> float:
        return float(self.cost_matrix[from_endpoint, to_endpoint])

    def straight_length(self, from_endpoint: int, to_endpoint: int) -> float:
        return float(self.straight_length_matrix[from_endpoint, to_endpoint])

    def should_snap(self, from_endpoint: int, to_endpoint: int) -> bool:
        return bool(self.snap_matrix[from_endpoint, to_endpoint])

    def connector_points_and_widths(
        self, from_endpoint: int, to_endpoint: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cache_key = (from_endpoint, to_endpoint)
        cached = self.raster_path_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.should_snap(from_endpoint, to_endpoint):
            node_path = shortest_graph_path(
                self.graph,
                self.endpoint_nodes[from_endpoint],
                self.endpoint_nodes[to_endpoint],
            )
            if node_path:
                points = self.graph_points[node_path]
                points, widths, _ = project_connector_to_strokes(
                    points,
                    self.endpoint_widths[from_endpoint],
                    self.endpoint_widths[to_endpoint],
                    self.segment_starts,
                    self.segment_ends,
                    self.segment_start_widths,
                    self.segment_end_widths,
                    self.raster_projection_distance,
                )
                result = (
                    points,
                    widths,
                    True,
                )
                self.raster_path_cache[cache_key] = result
                return result

        if self.raster_cost is not None:
            raster_points = raster_astar_connector(
                self.raster_cost,
                self.raster_line,
                self.raster_step,
                self.endpoint_positions[from_endpoint],
                self.endpoint_positions[to_endpoint],
                self.straight_length(from_endpoint, to_endpoint) * self.raster_max_ratio,
                self.raster_min_line_fraction,
            )
            if raster_points is not None:
                points, widths, projection_fraction = project_connector_to_strokes(
                    raster_points,
                    self.endpoint_widths[from_endpoint],
                    self.endpoint_widths[to_endpoint],
                    self.segment_starts,
                    self.segment_ends,
                    self.segment_start_widths,
                    self.segment_end_widths,
                    self.raster_projection_distance,
                )
                if projection_fraction >= self.raster_min_projection_fraction:
                    result = (points, widths, True)
                    self.raster_path_cache[cache_key] = result
                    return result

        points = self.endpoint_positions[[from_endpoint, to_endpoint]]
        widths = self.endpoint_widths[[from_endpoint, to_endpoint]]
        result = (points, widths, False)
        self.raster_path_cache[cache_key] = result
        return result


@dataclass
class ComponentRoute:
    component: int
    order: list[int]
    reversed_flags: list[bool]
    start_endpoint: int
    end_endpoint: int


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(gray.size)
    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_back = 0.0
    weight_back = 0.0
    best_threshold = 127
    best_variance = -1.0

    for threshold in range(256):
        weight_back += hist[threshold]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += threshold * hist[threshold]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        variance = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def load_grayscale(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def make_ink_mask(
    gray: np.ndarray, threshold: int | None = None, threshold_offset: int = 70
) -> tuple[np.ndarray, int, int]:
    otsu = otsu_threshold(gray)
    if threshold is None:
        threshold = int(np.clip(otsu + threshold_offset, 150, 220))
    return gray < threshold, otsu, threshold


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy, dx in NEIGHBORS_8:
        y0 = 1 + dy
        x0 = 1 + dx
        count += padded[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
    return count


def graph_neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy, dx in NEIGHBORS_8:
        neighbor = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        if dy != 0 and dx != 0:
            bridge_x = padded[1 : 1 + height, 1 + dx : 1 + dx + width]
            bridge_y = padded[1 + dy : 1 + dy + height, 1 : 1 + width]
            neighbor = neighbor & ~(bridge_x | bridge_y)
        count += neighbor
    return count


def zhang_suen_thin(mask: np.ndarray, max_iterations: int = 80) -> tuple[np.ndarray, int]:
    skeleton = mask.astype(bool).copy()

    def neighbors(image: np.ndarray) -> tuple[np.ndarray, ...]:
        padded = np.pad(image, 1, mode="constant", constant_values=False)
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]
        return p2, p3, p4, p5, p6, p7, p8, p9

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        before = int(skeleton.sum())
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(skeleton)
        count = (
            p2.astype(np.uint8)
            + p3.astype(np.uint8)
            + p4.astype(np.uint8)
            + p5.astype(np.uint8)
            + p6.astype(np.uint8)
            + p7.astype(np.uint8)
            + p8.astype(np.uint8)
            + p9.astype(np.uint8)
        )
        transitions = (
            (~p2 & p3).astype(np.uint8)
            + (~p3 & p4).astype(np.uint8)
            + (~p4 & p5).astype(np.uint8)
            + (~p5 & p6).astype(np.uint8)
            + (~p6 & p7).astype(np.uint8)
            + (~p7 & p8).astype(np.uint8)
            + (~p8 & p9).astype(np.uint8)
            + (~p9 & p2).astype(np.uint8)
        )
        remove = (
            skeleton
            & (count >= 2)
            & (count <= 6)
            & (transitions == 1)
            & ~(p2 & p4 & p6)
            & ~(p4 & p6 & p8)
        )
        skeleton[remove] = False

        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(skeleton)
        count = (
            p2.astype(np.uint8)
            + p3.astype(np.uint8)
            + p4.astype(np.uint8)
            + p5.astype(np.uint8)
            + p6.astype(np.uint8)
            + p7.astype(np.uint8)
            + p8.astype(np.uint8)
            + p9.astype(np.uint8)
        )
        transitions = (
            (~p2 & p3).astype(np.uint8)
            + (~p3 & p4).astype(np.uint8)
            + (~p4 & p5).astype(np.uint8)
            + (~p5 & p6).astype(np.uint8)
            + (~p6 & p7).astype(np.uint8)
            + (~p7 & p8).astype(np.uint8)
            + (~p8 & p9).astype(np.uint8)
            + (~p9 & p2).astype(np.uint8)
        )
        remove = (
            skeleton
            & (count >= 2)
            & (count <= 6)
            & (transitions == 1)
            & ~(p2 & p4 & p8)
            & ~(p2 & p6 & p8)
        )
        skeleton[remove] = False

        if int(skeleton.sum()) == before:
            break
    return skeleton, iterations


def edt_1d(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    locations = np.zeros(n, dtype=np.int32)
    boundaries = np.empty(n + 1, dtype=np.float64)
    distances = np.empty(n, dtype=np.float64)

    k = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for q in range(1, n):
        while True:
            p = locations[k]
            denominator = 2.0 * (q - p)
            if denominator == 0:
                intersection = np.inf
            else:
                intersection = ((values[q] + q * q) - (values[p] + p * p)) / denominator
            if intersection <= boundaries[k]:
                k -= 1
                if k < 0:
                    k = 0
                    break
            else:
                break
        k += 1
        locations[k] = q
        boundaries[k] = intersection
        boundaries[k + 1] = np.inf

    k = 0
    for q in range(n):
        while boundaries[k + 1] < q:
            k += 1
        p = locations[k]
        distances[q] = (q - p) * (q - p) + values[p]
    return distances


def distance_to_background(mask: np.ndarray) -> np.ndarray:
    # Felzenszwalb-Huttenlocher exact squared Euclidean distance transform.
    inf = float(mask.shape[0] * mask.shape[0] + mask.shape[1] * mask.shape[1])
    values = np.where(mask, inf, 0.0).astype(np.float64)
    temp = np.empty_like(values)
    for y in range(mask.shape[0]):
        temp[y, :] = edt_1d(values[y, :])
    dist2 = np.empty_like(values)
    for x in range(mask.shape[1]):
        dist2[:, x] = edt_1d(temp[:, x])
    return np.sqrt(dist2)


def erode_8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = mask.copy()
    for dy, dx in NEIGHBORS_8:
        y0 = 1 + dy
        x0 = 1 + dx
        eroded &= padded[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
    return eroded


def erosion_distance_to_background(mask: np.ndarray) -> np.ndarray:
    # A fast local width estimator. For clean line art it closely tracks the
    # half-width of each line and avoids the cost of a full Euclidean transform.
    distance = np.zeros(mask.shape, dtype=np.float64)
    current = mask.copy()
    radius = 0.0
    while current.any():
        radius += 1.0
        distance[current] = radius
        current = erode_8(current)
    return distance


def iter_skeleton_neighbors(skeleton: np.ndarray, y: int, x: int) -> Iterable[tuple[int, int]]:
    height, width = skeleton.shape
    for dy, dx in NEIGHBORS_8:
        yy = y + dy
        xx = x + dx
        if 0 <= yy < height and 0 <= xx < width and skeleton[yy, xx]:
            yield yy, xx


def iter_graph_neighbors(skeleton: np.ndarray, y: int, x: int) -> Iterable[tuple[int, int]]:
    height, width = skeleton.shape
    for dy, dx in NEIGHBORS_8:
        yy = y + dy
        xx = x + dx
        if not (0 <= yy < height and 0 <= xx < width and skeleton[yy, xx]):
            continue
        if dy != 0 and dx != 0:
            bridge_x = skeleton[y, xx]
            bridge_y = skeleton[yy, x]
            if bridge_x or bridge_y:
                continue
        yield yy, xx


def edge_key(width: int, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    aa = a[0] * width + a[1]
    bb = b[0] * width + b[1]
    return (aa, bb) if aa < bb else (bb, aa)


def trace_skeleton_paths(skeleton: np.ndarray) -> list[np.ndarray]:
    height, width = skeleton.shape
    degrees = graph_neighbor_count(skeleton)
    nodes = skeleton & (degrees != 2)
    visited_edges: set[tuple[int, int]] = set()
    paths: list[np.ndarray] = []

    def trace(start: tuple[int, int], second: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start]
        previous = start
        current = second
        max_steps = int(skeleton.sum()) + 5

        for _ in range(max_steps):
            visited_edges.add(edge_key(width, previous, current))
            path.append(current)
            if nodes[current] and current != start:
                break

            next_candidates = [
                point for point in iter_graph_neighbors(skeleton, *current) if point != previous
            ]
            unvisited = [
                point
                for point in next_candidates
                if edge_key(width, current, point) not in visited_edges
            ]
            if not unvisited:
                break
            if len(unvisited) == 1:
                following = unvisited[0]
            else:
                # In rare 2x2 skeleton artifacts, continue as straight as possible.
                py, px = previous
                cy, cx = current
                incoming = np.array([cy - py, cx - px], dtype=np.float64)
                best_score = -np.inf
                following = unvisited[0]
                for candidate in unvisited:
                    ny, nx = candidate
                    outgoing = np.array([ny - cy, nx - cx], dtype=np.float64)
                    score = float(np.dot(incoming, outgoing))
                    if score > best_score:
                        best_score = score
                        following = candidate
            previous, current = current, following
            if current == start:
                visited_edges.add(edge_key(width, previous, current))
                path.append(current)
                break
        return path

    ys, xs = np.nonzero(nodes)
    for y, x in zip(ys.tolist(), xs.tolist()):
        start = (y, x)
        for neighbor in iter_graph_neighbors(skeleton, y, x):
            key = edge_key(width, start, neighbor)
            if key not in visited_edges:
                raw_path = trace(start, neighbor)
                if len(raw_path) > 1:
                    paths.append(points_yx_to_xy(raw_path))

    ys, xs = np.nonzero(skeleton)
    for y, x in zip(ys.tolist(), xs.tolist()):
        start = (y, x)
        for neighbor in iter_graph_neighbors(skeleton, y, x):
            key = edge_key(width, start, neighbor)
            if key not in visited_edges:
                raw_path = trace(start, neighbor)
                if len(raw_path) > 1:
                    paths.append(points_yx_to_xy(raw_path))

    return paths


def points_yx_to_xy(points: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([(x, y) for y, x in points], dtype=np.float64)


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.sqrt((deltas * deltas).sum(axis=1)).sum())


def reorder_closed_path(points: np.ndarray) -> np.ndarray:
    if len(points) < 4 or np.linalg.norm(points[0] - points[-1]) > 0.01:
        return points
    loop = points[:-1]
    opposite = int(np.argmax(((loop - loop[0]) ** 2).sum(axis=1)))
    reordered = np.concatenate([loop[opposite:], loop[: opposite + 1]], axis=0)
    return reordered


def smooth_polyline(points: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or len(points) < 4:
        return points

    closed = np.linalg.norm(points[0] - points[-1]) <= 0.01
    body = points[:-1].copy() if closed else points.copy()
    if len(body) < 4:
        return points

    for _ in range(passes):
        if closed:
            body = 0.25 * np.roll(body, 1, axis=0) + 0.5 * body + 0.25 * np.roll(
                body, -1, axis=0
            )
        else:
            smoothed = body.copy()
            smoothed[1:-1] = 0.25 * body[:-2] + 0.5 * body[1:-1] + 0.25 * body[2:]
            body = smoothed

    if closed:
        return np.vstack([body, body[0]])
    return body


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2 or epsilon <= 0:
        return points

    points = reorder_closed_path(points)
    if len(points) <= 2:
        return points

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a = points[start]
        b = points[end]
        segment = b - a
        segment_length = float(np.linalg.norm(segment))
        middle = points[start + 1 : end]
        if segment_length == 0:
            distances = np.sqrt(((middle - a) ** 2).sum(axis=1))
        else:
            relative = middle - a
            distances = np.abs(segment[0] * relative[:, 1] - segment[1] * relative[:, 0])
            distances /= segment_length
        max_index = int(np.argmax(distances))
        max_distance = float(distances[max_index])
        if max_distance > epsilon:
            split = start + 1 + max_index
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))

    return points[keep]


def sample_widths(points: np.ndarray, width_map: np.ndarray, min_width: float = 1.0) -> np.ndarray:
    height, width = width_map.shape
    xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    sampled = 2.0 * width_map[ys, xs]
    sampled = np.maximum(sampled, min_width)
    return sampled.astype(np.float64)


def build_strokes(
    skeleton: np.ndarray,
    width_map: np.ndarray,
    simplify_epsilon: float,
    min_length: float,
    min_width: float,
    smooth_passes: int,
) -> list[Stroke]:
    raw_paths = trace_skeleton_paths(skeleton)
    strokes: list[Stroke] = []
    for path in raw_paths:
        smoothed = smooth_polyline(path, smooth_passes)
        simplified = rdp_simplify(smoothed, simplify_epsilon)
        if len(simplified) < 2:
            continue
        widths = sample_widths(simplified, width_map, min_width=min_width)
        length = polyline_length(simplified)
        if length < min_length:
            continue
        strokes.append(
            Stroke(
                points=simplified,
                widths=widths,
                length=length,
                median_width=float(np.median(widths)),
            )
        )
    strokes.sort(key=lambda stroke: stroke.length, reverse=True)
    return strokes


def stroke_endpoint_ids(stroke_index: int, reversed_flag: bool) -> tuple[int, int]:
    start_id = stroke_index * 2
    end_id = start_id + 1
    return (end_id, start_id) if reversed_flag else (start_id, end_id)


def endpoint_positions_and_widths(strokes: Sequence[Stroke]) -> tuple[np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    widths: list[float] = []
    for stroke in strokes:
        positions.append(stroke.start)
        positions.append(stroke.end)
        widths.append(float(stroke.widths[0]))
        widths.append(float(stroke.widths[-1]))
    if not positions:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(positions, dtype=np.float64), np.asarray(widths, dtype=np.float64)


def quantized_point_key(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]) * 1000.0)), int(round(float(point[1]) * 1000.0))


def build_stroke_graph(
    strokes: Sequence[Stroke],
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, float]]], list[int]]:
    node_by_key: dict[tuple[int, int], int] = {}
    points: list[np.ndarray] = []
    width_sums: list[float] = []
    width_counts: list[int] = []
    edge_weights: dict[tuple[int, int], float] = {}

    def node_id(point: np.ndarray, width: float) -> int:
        key = quantized_point_key(point)
        existing = node_by_key.get(key)
        if existing is not None:
            width_sums[existing] += float(width)
            width_counts[existing] += 1
            return existing
        idx = len(points)
        node_by_key[key] = idx
        points.append(np.asarray(point, dtype=np.float64))
        width_sums.append(float(width))
        width_counts.append(1)
        return idx

    for stroke in strokes:
        previous: int | None = None
        for point, width in zip(stroke.points, stroke.widths):
            current = node_id(point, float(width))
            if previous is not None and previous != current:
                length = float(np.linalg.norm(points[current] - points[previous]))
                key = (previous, current) if previous < current else (current, previous)
                old = edge_weights.get(key)
                if old is None or length < old:
                    edge_weights[key] = length
            previous = current

    graph: list[list[tuple[int, float]]] = [[] for _ in points]
    for (a, b), length in edge_weights.items():
        graph[a].append((b, length))
        graph[b].append((a, length))

    endpoint_nodes: list[int] = []
    for stroke in strokes:
        endpoint_nodes.append(node_id(stroke.start, float(stroke.widths[0])))
        endpoint_nodes.append(node_id(stroke.end, float(stroke.widths[-1])))

    node_points = np.asarray(points, dtype=np.float64)
    node_widths = np.asarray(width_sums, dtype=np.float64) / np.maximum(
        np.asarray(width_counts, dtype=np.float64), 1.0
    )
    return node_points, node_widths, graph, endpoint_nodes


def build_stroke_segments(
    strokes: Sequence[Stroke],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    start_widths: list[float] = []
    end_widths: list[float] = []
    for stroke in strokes:
        for i in range(len(stroke.points) - 1):
            a = stroke.points[i]
            b = stroke.points[i + 1]
            if float(np.linalg.norm(b - a)) <= 0:
                continue
            starts.append(a)
            ends.append(b)
            start_widths.append(float(stroke.widths[i]))
            end_widths.append(float(stroke.widths[i + 1]))
    if not starts:
        empty_points = np.empty((0, 2), dtype=np.float64)
        empty_widths = np.empty((0,), dtype=np.float64)
        return empty_points, empty_points, empty_widths, empty_widths
    return (
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
        np.asarray(start_widths, dtype=np.float64),
        np.asarray(end_widths, dtype=np.float64),
    )


def project_point_to_segments(
    point: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_start_widths: np.ndarray,
    segment_end_widths: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, float, bool]:
    if len(segment_starts) == 0:
        return point, 1.0, False
    segments = segment_ends - segment_starts
    lengths2 = np.maximum((segments * segments).sum(axis=1), 1e-9)
    relative = point[None, :] - segment_starts
    t = np.clip((relative * segments).sum(axis=1) / lengths2, 0.0, 1.0)
    projections = segment_starts + segments * t[:, None]
    distances2 = ((projections - point[None, :]) ** 2).sum(axis=1)
    best = int(np.argmin(distances2))
    distance = math.sqrt(float(distances2[best]))
    if distance > max_distance:
        return point, 1.0, False
    width = segment_start_widths[best] * (1.0 - t[best]) + segment_end_widths[best] * t[best]
    return projections[best], float(width), True


def project_connector_to_strokes(
    points: np.ndarray,
    start_width: float,
    end_width: float,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_start_widths: np.ndarray,
    segment_end_widths: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) <= 2 or len(segment_starts) == 0 or max_distance <= 0:
        return points, sample_endpoint_connector_widths(points, start_width, end_width), 0.0

    fallback_widths = sample_endpoint_connector_widths(points, start_width, end_width)
    projected_points = [np.asarray(points[0], dtype=np.float64)]
    projected_widths = [float(start_width)]
    snapped_flags = [True]
    for point, fallback_width in zip(points[1:-1], fallback_widths[1:-1]):
        projected, width, snapped = project_point_to_segments(
            point,
            segment_starts,
            segment_ends,
            segment_start_widths,
            segment_end_widths,
            max_distance,
        )
        projected_points.append(projected)
        projected_widths.append(width if snapped else float(fallback_width))
        snapped_flags.append(snapped)
    projected_points.append(np.asarray(points[-1], dtype=np.float64))
    projected_widths.append(float(end_width))
    snapped_flags.append(True)

    original_lengths = np.sqrt(((np.diff(points, axis=0)) ** 2).sum(axis=1))
    total_length = float(original_lengths.sum())
    if total_length > 0:
        segment_flags = np.asarray(snapped_flags[:-1]) & np.asarray(snapped_flags[1:])
        projection_fraction = float(original_lengths[segment_flags].sum() / total_length)
    else:
        projection_fraction = 1.0

    compact_points: list[np.ndarray] = []
    compact_widths: list[float] = []
    for point, width in zip(projected_points, projected_widths):
        if compact_points and float(np.linalg.norm(point - compact_points[-1])) < 0.35:
            compact_points[-1] = point
            compact_widths[-1] = width
        else:
            compact_points.append(point)
            compact_widths.append(width)

    return (
        np.asarray(compact_points, dtype=np.float64),
        np.asarray(compact_widths, dtype=np.float64),
        projection_fraction,
    )


def dijkstra_distances(graph: Sequence[Sequence[tuple[int, float]]], start: int) -> list[float]:
    distances = [math.inf] * len(graph)
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_cost = cost + weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
    return distances


def shortest_graph_path(
    graph: Sequence[Sequence[tuple[int, float]]], start: int, goal: int
) -> list[int]:
    if start == goal:
        return [start]
    distances = [math.inf] * len(graph)
    previous = [-1] * len(graph)
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == goal:
            break
        for neighbor, weight in graph[node]:
            new_cost = cost + weight
            if new_cost < distances[neighbor]:
                previous[neighbor] = node
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))

    if not math.isfinite(distances[goal]):
        return []

    path = [goal]
    node = goal
    while node != start:
        node = previous[node]
        if node < 0:
            return []
        path.append(node)
    path.reverse()
    return path


def bounds_from_strokes(strokes: Sequence[Stroke], size: tuple[int, int]) -> tuple[float, float, float, float]:
    if not strokes:
        width, height = size
        return 0.0, 0.0, float(width), float(height)
    points = np.concatenate([stroke.points for stroke in strokes], axis=0)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])


def face_avoidance_values(points: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    # Portrait-specific soft mask: strong in the eye/nose/mouth area, weaker
    # around the lower face and neck. It is intentionally smooth so route
    # changes are stable under small image differences.
    face_center = np.asarray([min_x + 0.50 * width, min_y + 0.50 * height])
    face_radius = np.asarray([0.24 * width, 0.25 * height])
    face_delta = (points - face_center) / face_radius
    face = np.clip(1.0 - np.sum(face_delta * face_delta, axis=1), 0.0, 1.0)

    neck_center = np.asarray([min_x + 0.52 * width, min_y + 0.74 * height])
    neck_radius = np.asarray([0.15 * width, 0.16 * height])
    neck_delta = (points - neck_center) / neck_radius
    neck = 0.55 * np.clip(1.0 - np.sum(neck_delta * neck_delta, axis=1), 0.0, 1.0)

    return np.maximum(face, neck)


def weighted_straight_cost(
    start: np.ndarray,
    end: np.ndarray,
    bounds: tuple[float, float, float, float],
    avoid_strength: float,
) -> tuple[float, float]:
    length = float(np.linalg.norm(end - start))
    if length == 0:
        return 0.0, 0.0
    if avoid_strength <= 0:
        return length, length
    samples = np.linspace(0.1, 0.9, 9, dtype=np.float64)[:, None]
    points = start[None, :] + (end - start)[None, :] * samples
    avoid = float(face_avoidance_values(points, bounds).mean())
    return length * (1.0 + avoid_strength * avoid), length


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        dilated = out.copy()
        for dy, dx in NEIGHBORS_8:
            dilated |= padded[
                1 + dy : 1 + dy + mask.shape[0],
                1 + dx : 1 + dx + mask.shape[1],
            ]
        out = dilated
    return out


def downsample_mask_max(mask: np.ndarray, step: int) -> np.ndarray:
    step = max(1, int(step))
    height, width = mask.shape
    small_width = int(math.ceil(width / step))
    small_height = int(math.ceil(height / step))
    padded = np.zeros((small_height * step, small_width * step), dtype=bool)
    padded[:height, :width] = mask
    blocks = padded.reshape(small_height, step, small_width, step)
    return blocks.max(axis=(1, 3))


def build_raster_cost_map(
    ink: np.ndarray,
    strokes: Sequence[Stroke],
    size: tuple[int, int],
    step: int,
    avoid_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, int(step))
    line_mask = dilate_mask(ink, max(1, int(round(5 / step))))
    small_line = downsample_mask_max(line_mask, step)
    height, width = small_line.shape
    cost = np.full((height, width), 2.0, dtype=np.float64)
    cost[small_line] = 0.22

    if avoid_strength > 0:
        bounds = bounds_from_strokes(strokes, size)
        xs = (np.arange(width, dtype=np.float64) + 0.5) * step
        ys = (np.arange(height, dtype=np.float64) + 0.5) * step
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        avoid = face_avoidance_values(points, bounds).reshape(height, width)
        cost += avoid * avoid_strength * 2.0
        cost[small_line] *= 0.45
    return cost, small_line


def sample_endpoint_connector_widths(
    points: np.ndarray, start_width: float, end_width: float
) -> np.ndarray:
    if len(points) <= 1:
        return np.asarray([start_width], dtype=np.float64)
    distances = np.zeros(len(points), dtype=np.float64)
    distances[1:] = np.sqrt(((np.diff(points, axis=0)) ** 2).sum(axis=1)).cumsum()
    total = distances[-1]
    if total <= 0:
        return np.full(len(points), (start_width + end_width) * 0.5, dtype=np.float64)
    t = distances / total
    return start_width * (1.0 - t) + end_width * t


def simplify_connector_points(points: np.ndarray, epsilon: float = 1.5) -> np.ndarray:
    if len(points) <= 2:
        return points
    return rdp_simplify(points, epsilon)


def raster_astar_connector(
    cost_map: np.ndarray,
    line_mask: np.ndarray | None,
    step: int,
    start: np.ndarray,
    goal: np.ndarray,
    max_physical_length: float,
    min_line_fraction: float,
) -> np.ndarray | None:
    if max_physical_length <= 0:
        return None
    height, width = cost_map.shape
    start_cell = (
        int(np.clip(round(float(start[1]) / step), 0, height - 1)),
        int(np.clip(round(float(start[0]) / step), 0, width - 1)),
    )
    goal_cell = (
        int(np.clip(round(float(goal[1]) / step), 0, height - 1)),
        int(np.clip(round(float(goal[0]) / step), 0, width - 1)),
    )
    if start_cell == goal_cell:
        return np.vstack([start, goal])

    max_steps_length = max_physical_length / step
    min_cost = max(float(cost_map.min()), 0.05)

    def heuristic(cell: tuple[int, int]) -> float:
        dy = cell[0] - goal_cell[0]
        dx = cell[1] - goal_cell[1]
        return math.sqrt(dx * dx + dy * dy) * min_cost

    distances: dict[tuple[int, int], float] = {start_cell: 0.0}
    physical_lengths: dict[tuple[int, int], float] = {start_cell: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, tuple[int, int]]] = [(heuristic(start_cell), start_cell)]
    closed: set[tuple[int, int]] = set()

    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    while queue:
        _, cell = heapq.heappop(queue)
        if cell in closed:
            continue
        if cell == goal_cell:
            break
        closed.add(cell)
        base_cost = distances[cell]
        base_length = physical_lengths[cell]
        for dy, dx, move_length in moves:
            yy = cell[0] + dy
            xx = cell[1] + dx
            if yy < 0 or yy >= height or xx < 0 or xx >= width:
                continue
            neighbor = (yy, xx)
            physical_length = base_length + move_length
            if physical_length > max_steps_length:
                continue
            step_cost = move_length * 0.5 * (cost_map[cell] + cost_map[neighbor])
            new_cost = base_cost + float(step_cost)
            if new_cost < distances.get(neighbor, math.inf):
                distances[neighbor] = new_cost
                physical_lengths[neighbor] = physical_length
                previous[neighbor] = cell
                heapq.heappush(queue, (new_cost + heuristic(neighbor), neighbor))

    if goal_cell not in previous:
        return None

    cells = [goal_cell]
    cell = goal_cell
    while cell != start_cell:
        cell = previous[cell]
        cells.append(cell)
    cells.reverse()

    if line_mask is not None and min_line_fraction > 0:
        on_line = [line_mask[y, x] for y, x in cells]
        if float(np.mean(on_line)) < min_line_fraction:
            return None

    points = np.asarray(
        [[x * step, y * step] for y, x in cells],
        dtype=np.float64,
    )
    points[0] = start
    points[-1] = goal
    physical_length = polyline_length(points)
    if physical_length > max_physical_length:
        return None
    return points


def build_connector_model(
    strokes: Sequence[Stroke],
    size: tuple[int, int],
    ink: np.ndarray | None,
    avoid_strength: float,
    retrace_weight: float,
    snap_max_ratio: float,
    snap_cost_ratio: float,
    raster_step: int,
    raster_max_ratio: float,
    raster_min_line_fraction: float,
    raster_projection_distance: float,
    raster_min_projection_fraction: float,
) -> ConnectorModel | None:
    if not strokes:
        return None

    endpoint_positions, endpoint_widths = endpoint_positions_and_widths(strokes)
    endpoint_count = len(endpoint_positions)
    if endpoint_count == 0:
        return None

    graph_points, graph_widths, graph, endpoint_nodes = build_stroke_graph(strokes)
    segment_starts, segment_ends, segment_start_widths, segment_end_widths = build_stroke_segments(
        strokes
    )
    bounds = bounds_from_strokes(strokes, size)

    straight_cost = np.zeros((endpoint_count, endpoint_count), dtype=np.float64)
    straight_length = np.zeros((endpoint_count, endpoint_count), dtype=np.float64)
    for i in range(endpoint_count):
        for j in range(i + 1, endpoint_count):
            cost, length = weighted_straight_cost(
                endpoint_positions[i], endpoint_positions[j], bounds, avoid_strength
            )
            straight_cost[i, j] = straight_cost[j, i] = cost
            straight_length[i, j] = straight_length[j, i] = length

    graph_length = np.full((endpoint_count, endpoint_count), math.inf, dtype=np.float64)
    for i, node in enumerate(endpoint_nodes):
        distances = dijkstra_distances(graph, node)
        for j, other_node in enumerate(endpoint_nodes):
            graph_length[i, j] = distances[other_node]

    snap_matrix = np.zeros((endpoint_count, endpoint_count), dtype=bool)
    cost_matrix = straight_cost.copy()
    retrace_weight = max(0.0, retrace_weight)
    snap_max_ratio = max(1.0, snap_max_ratio)
    snap_cost_ratio = max(0.0, snap_cost_ratio)
    for i in range(endpoint_count):
        for j in range(endpoint_count):
            if i == j:
                cost_matrix[i, j] = 0.0
                graph_length[i, j] = 0.0
                continue
            path_length = graph_length[i, j]
            direct_length = straight_length[i, j]
            if not math.isfinite(path_length) or direct_length <= 0:
                continue
            retrace_cost = path_length * retrace_weight
            allowed = path_length <= direct_length * snap_max_ratio
            cheaper = retrace_cost <= straight_cost[i, j] * snap_cost_ratio
            if allowed and cheaper:
                snap_matrix[i, j] = True
                cost_matrix[i, j] = retrace_cost

    raster_cost: np.ndarray | None = None
    raster_line: np.ndarray | None = None
    if ink is not None and raster_max_ratio > 1.0:
        raster_cost, raster_line = build_raster_cost_map(
            ink, strokes, size, raster_step, avoid_strength
        )

    return ConnectorModel(
        endpoint_positions=endpoint_positions,
        endpoint_widths=endpoint_widths,
        cost_matrix=cost_matrix,
        straight_cost_matrix=straight_cost,
        straight_length_matrix=straight_length,
        graph_length_matrix=graph_length,
        snap_matrix=snap_matrix,
        graph_points=graph_points,
        graph_widths=graph_widths,
        graph=graph,
        endpoint_nodes=endpoint_nodes,
        retrace_weight=retrace_weight,
        segment_starts=segment_starts,
        segment_ends=segment_ends,
        segment_start_widths=segment_start_widths,
        segment_end_widths=segment_end_widths,
        raster_cost=raster_cost,
        raster_line=raster_line,
        raster_step=max(1, raster_step),
        raster_max_ratio=max(1.0, raster_max_ratio),
        raster_min_line_fraction=max(0.0, raster_min_line_fraction),
        raster_projection_distance=max(0.0, raster_projection_distance),
        raster_min_projection_fraction=max(0.0, raster_min_projection_fraction),
    )


def build_stroke_endpoint_graph(
    strokes: Sequence[Stroke],
) -> tuple[np.ndarray, list[tuple[int, int]], list[list[tuple[int, int]]]]:
    node_by_key: dict[tuple[int, int], int] = {}
    node_points: list[np.ndarray] = []
    stroke_nodes: list[tuple[int, int]] = []

    def node_id(point: np.ndarray) -> int:
        key = quantized_point_key(point)
        existing = node_by_key.get(key)
        if existing is not None:
            return existing
        idx = len(node_points)
        node_by_key[key] = idx
        node_points.append(np.asarray(point, dtype=np.float64))
        return idx

    for stroke in strokes:
        stroke_nodes.append((node_id(stroke.start), node_id(stroke.end)))

    adjacency: list[list[tuple[int, int]]] = [[] for _ in node_points]
    for stroke_id, (start_node, end_node) in enumerate(stroke_nodes):
        adjacency[start_node].append((stroke_id, end_node))
        if end_node != start_node:
            adjacency[end_node].append((stroke_id, start_node))
    return np.asarray(node_points, dtype=np.float64), stroke_nodes, adjacency


def stroke_components(
    stroke_nodes: Sequence[tuple[int, int]], adjacency: Sequence[Sequence[tuple[int, int]]]
) -> list[tuple[set[int], set[int]]]:
    remaining = set(range(len(stroke_nodes)))
    components: list[tuple[set[int], set[int]]] = []
    while remaining:
        first = next(iter(remaining))
        edge_ids: set[int] = set()
        node_ids: set[int] = set()
        stack = [stroke_nodes[first][0], stroke_nodes[first][1]]
        while stack:
            node = stack.pop()
            if node in node_ids:
                continue
            node_ids.add(node)
            for stroke_id, other_node in adjacency[node]:
                if stroke_id not in edge_ids:
                    edge_ids.add(stroke_id)
                    stack.extend(stroke_nodes[stroke_id])
                if other_node not in node_ids:
                    stack.append(other_node)
        remaining.difference_update(edge_ids)
        components.append((edge_ids, node_ids))
    components.sort(key=lambda item: len(item[0]), reverse=True)
    return components


def component_start_nodes(
    node_points: np.ndarray,
    edge_ids: set[int],
    node_ids: set[int],
    stroke_nodes: Sequence[tuple[int, int]],
) -> list[int]:
    degree = {node: 0 for node in node_ids}
    for stroke_id in edge_ids:
        a, b = stroke_nodes[stroke_id]
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    leaves = [node for node, count in degree.items() if count <= 1]
    candidates = leaves if leaves else list(node_ids)

    def ranked(nodes: Sequence[int], key_fn) -> list[int]:
        ordered = sorted(nodes, key=lambda node: key_fn(node_points[node]))
        return [ordered[0], ordered[-1]] if len(ordered) > 1 else ordered

    starts: list[int] = []
    for key_fn in (
        lambda point: point[0] + point[1],
        lambda point: point[0] - point[1],
        lambda point: point[0],
        lambda point: point[1],
    ):
        starts.extend(ranked(candidates, key_fn))
    if candidates:
        starts.append(max(candidates, key=lambda node: degree.get(node, 0)))

    deduped: list[int] = []
    seen: set[int] = set()
    for node in starts:
        if node not in seen:
            deduped.append(node)
            seen.add(node)
    return deduped[:10]


def stroke_flag_from_node(
    stroke_id: int, from_node: int, stroke_nodes: Sequence[tuple[int, int]]
) -> bool:
    start_node, end_node = stroke_nodes[stroke_id]
    if from_node == start_node:
        return False
    if from_node == end_node:
        return True
    raise ValueError("stroke is not incident to node")


def reverse_component_route(route: ComponentRoute) -> ComponentRoute:
    return ComponentRoute(
        component=route.component,
        order=list(reversed(route.order)),
        reversed_flags=[not flag for flag in reversed(route.reversed_flags)],
        start_endpoint=route.end_endpoint,
        end_endpoint=route.start_endpoint,
    )


def endpoint_for_node_side(
    stroke_id: int, node: int, stroke_nodes: Sequence[tuple[int, int]]
) -> int:
    start_node, end_node = stroke_nodes[stroke_id]
    if node == start_node:
        return stroke_id * 2
    if node == end_node:
        return stroke_id * 2 + 1
    raise ValueError("stroke is not incident to node")


def traverse_component_edges(
    component_id: int,
    edge_ids: set[int],
    start_node: int,
    stroke_nodes: Sequence[tuple[int, int]],
    adjacency: Sequence[Sequence[tuple[int, int]]],
    strokes: Sequence[Stroke],
) -> ComponentRoute | None:
    seen_edges: set[int] = set()
    order: list[int] = []
    reversed_flags: list[bool] = []
    first_start_endpoint: int | None = None
    last_end_endpoint: int | None = None

    sorted_adjacency: dict[int, list[tuple[int, int]]] = {}
    for node, entries in enumerate(adjacency):
        filtered = [entry for entry in entries if entry[0] in edge_ids]
        filtered.sort(key=lambda entry: strokes[entry[0]].length, reverse=True)
        sorted_adjacency[node] = filtered

    def append_stroke(stroke_id: int, from_node: int) -> int:
        nonlocal first_start_endpoint, last_end_endpoint
        flag = stroke_flag_from_node(stroke_id, from_node, stroke_nodes)
        start_endpoint, end_endpoint = stroke_endpoint_ids(stroke_id, flag)
        if first_start_endpoint is None:
            first_start_endpoint = start_endpoint
        last_end_endpoint = end_endpoint
        order.append(stroke_id)
        reversed_flags.append(flag)
        a, b = stroke_nodes[stroke_id]
        return b if from_node == a else a

    def dfs(node: int) -> int:
        current = node
        for stroke_id, other_node in sorted_adjacency[node]:
            if stroke_id in seen_edges:
                continue
            seen_edges.add(stroke_id)
            current = append_stroke(stroke_id, node)
            current = dfs(other_node)
            if len(seen_edges) < len(edge_ids):
                current = append_stroke(stroke_id, other_node)
        return current

    dfs(start_node)
    if not order or first_start_endpoint is None or last_end_endpoint is None:
        return None
    return ComponentRoute(
        component=component_id,
        order=order,
        reversed_flags=reversed_flags,
        start_endpoint=first_start_endpoint,
        end_endpoint=last_end_endpoint,
    )


def component_route_candidates(strokes: Sequence[Stroke]) -> list[list[ComponentRoute]]:
    if not strokes:
        return []
    node_points, stroke_nodes, adjacency = build_stroke_endpoint_graph(strokes)
    components = stroke_components(stroke_nodes, adjacency)
    all_candidates: list[list[ComponentRoute]] = []
    for component_id, (edge_ids, node_ids) in enumerate(components):
        candidates: list[ComponentRoute] = []
        for start_node in component_start_nodes(node_points, edge_ids, node_ids, stroke_nodes):
            route = traverse_component_edges(
                component_id, edge_ids, start_node, stroke_nodes, adjacency, strokes
            )
            if route is None:
                continue
            candidates.append(route)
            candidates.append(reverse_component_route(route))

        deduped: list[ComponentRoute] = []
        seen: set[tuple[tuple[int, ...], tuple[bool, ...]]] = set()
        for route in candidates:
            key = (tuple(route.order), tuple(route.reversed_flags))
            if key not in seen:
                deduped.append(route)
                seen.add(key)
        if deduped:
            all_candidates.append(deduped)
    return all_candidates


def component_route_cost(
    strokes: Sequence[Stroke],
    selected: Sequence[ComponentRoute],
    connector_model: ConnectorModel | None,
) -> float:
    if len(selected) < 2:
        return 0.0
    total = 0.0
    previous_end = selected[0].end_endpoint
    for route in selected[1:]:
        total += connector_cost(strokes, connector_model, previous_end, route.start_endpoint)
        previous_end = route.end_endpoint
    return total


def solve_component_walk_route(
    strokes: Sequence[Stroke], connector_model: ConnectorModel | None = None
) -> Route:
    candidates_by_component = component_route_candidates(strokes)
    if not candidates_by_component:
        return Route([], [], 0.0)

    best_selection: list[ComponentRoute] | None = None
    best_cost = math.inf
    seed_limit = min(len(candidates_by_component), 8)
    seed_pairs: list[tuple[int, int]] = []
    for component_idx in range(seed_limit):
        for candidate_idx in range(min(len(candidates_by_component[component_idx]), 6)):
            seed_pairs.append((component_idx, candidate_idx))

    for seed_component_idx, seed_candidate_idx in seed_pairs:
        seed = candidates_by_component[seed_component_idx][seed_candidate_idx]
        unused = set(range(len(candidates_by_component)))
        unused.remove(seed_component_idx)
        selected = [seed]
        current_end = seed.end_endpoint

        while unused:
            best_next: tuple[float, int, ComponentRoute] | None = None
            for component_idx in unused:
                for candidate in candidates_by_component[component_idx]:
                    cost = connector_cost(
                        strokes, connector_model, current_end, candidate.start_endpoint
                    )
                    if best_next is None or cost < best_next[0]:
                        best_next = (cost, component_idx, candidate)
            assert best_next is not None
            _, component_idx, candidate = best_next
            selected.append(candidate)
            current_end = candidate.end_endpoint
            unused.remove(component_idx)

        cost = component_route_cost(strokes, selected, connector_model)
        if cost < best_cost:
            best_cost = cost
            best_selection = selected

    assert best_selection is not None
    order: list[int] = []
    reversed_flags: list[bool] = []
    for component_route in best_selection:
        order.extend(component_route.order)
        reversed_flags.extend(component_route.reversed_flags)
    return Route(
        order=order,
        reversed_flags=reversed_flags,
        connector_length=route_cost(strokes, order, reversed_flags, connector_model),
    )


def oriented_start_end(stroke: Stroke, reversed_flag: bool) -> tuple[np.ndarray, np.ndarray]:
    if reversed_flag:
        return stroke.end, stroke.start
    return stroke.start, stroke.end


def connector_cost(
    strokes: Sequence[Stroke],
    connector_model: ConnectorModel | None,
    from_endpoint: int,
    to_endpoint: int,
) -> float:
    if connector_model is not None:
        return connector_model.cost(from_endpoint, to_endpoint)
    from_stroke = strokes[from_endpoint // 2]
    to_stroke = strokes[to_endpoint // 2]
    from_point = from_stroke.start if from_endpoint % 2 == 0 else from_stroke.end
    to_point = to_stroke.start if to_endpoint % 2 == 0 else to_stroke.end
    return float(np.linalg.norm(from_point - to_point))


def route_cost(
    strokes: Sequence[Stroke],
    order: Sequence[int],
    reversed_flags: Sequence[bool],
    connector_model: ConnectorModel | None = None,
) -> float:
    if len(order) < 2:
        return 0.0
    total = 0.0
    _, previous_end_id = stroke_endpoint_ids(order[0], reversed_flags[0])
    for idx, reversed_flag in zip(order[1:], reversed_flags[1:]):
        start_id, end_id = stroke_endpoint_ids(idx, reversed_flag)
        total += connector_cost(strokes, connector_model, previous_end_id, start_id)
        previous_end_id = end_id
    return total


def nearest_neighbor_route(
    strokes: Sequence[Stroke],
    start_index: int,
    start_reversed: bool,
    connector_model: ConnectorModel | None = None,
) -> Route:
    n = len(strokes)
    endpoints = np.asarray([[stroke.start, stroke.end] for stroke in strokes], dtype=np.float64)
    unused = np.ones(n, dtype=bool)
    order = [start_index]
    reversed_flags = [start_reversed]
    unused[start_index] = False
    _, current_end_id = stroke_endpoint_ids(start_index, start_reversed)
    current_end = endpoints[start_index, 0 if start_reversed else 1]

    for _ in range(n - 1):
        remaining = np.flatnonzero(unused)
        if connector_model is None:
            starts = endpoints[remaining, 0]
            ends = endpoints[remaining, 1]
            dist_to_start = ((starts - current_end) ** 2).sum(axis=1)
            dist_to_end = ((ends - current_end) ** 2).sum(axis=1)
        else:
            dist_to_start = connector_model.cost_matrix[current_end_id, remaining * 2]
            dist_to_end = connector_model.cost_matrix[current_end_id, remaining * 2 + 1]
        use_end = dist_to_end < dist_to_start
        best_dist = np.where(use_end, dist_to_end, dist_to_start)
        choice_position = int(np.argmin(best_dist))
        choice = int(remaining[choice_position])
        reversed_flag = bool(use_end[choice_position])
        order.append(choice)
        reversed_flags.append(reversed_flag)
        unused[choice] = False
        _, current_end_id = stroke_endpoint_ids(choice, reversed_flag)
        current_end = endpoints[choice, 0 if reversed_flag else 1]

    return Route(order, reversed_flags, route_cost(strokes, order, reversed_flags, connector_model))


def endpoint_seed_indices(strokes: Sequence[Stroke]) -> list[tuple[int, bool]]:
    if not strokes:
        return []
    starts = np.asarray([stroke.start for stroke in strokes])
    ends = np.asarray([stroke.end for stroke in strokes])
    endpoints = np.stack([starts, ends], axis=1)
    flat = endpoints.reshape(-1, 2)
    seed_positions = {
        int(np.argmin(flat[:, 0] + flat[:, 1])),
        int(np.argmax(flat[:, 0] + flat[:, 1])),
        int(np.argmin(flat[:, 0] - flat[:, 1])),
        int(np.argmax(flat[:, 0] - flat[:, 1])),
        0,
        1 if len(flat) > 1 else 0,
    }

    seeds: list[tuple[int, bool]] = []
    for flat_position in seed_positions:
        stroke_index = flat_position // 2
        endpoint_index = flat_position % 2
        seeds.append((stroke_index, endpoint_index == 1))

    longest_count = min(5, len(strokes))
    for stroke_index in range(longest_count):
        seeds.append((stroke_index, False))
        seeds.append((stroke_index, True))

    deduped: list[tuple[int, bool]] = []
    seen: set[tuple[int, bool]] = set()
    for seed in seeds:
        if seed not in seen:
            deduped.append(seed)
            seen.add(seed)
    return deduped


def improve_orientations(
    strokes: Sequence[Stroke],
    route: Route,
    connector_model: ConnectorModel | None = None,
    fixed_start: bool = False,
) -> bool:
    improved = False
    flags = route.reversed_flags
    order = route.order
    n = len(order)
    first_mutable = 1 if fixed_start else 0
    for i in range(first_mutable, n):
        old_flag = flags[i]

        def local_cost(flag: bool) -> float:
            flags[i] = flag
            start_id, end_id = stroke_endpoint_ids(order[i], flags[i])
            cost = 0.0
            if i > 0:
                _, prev_end_id = stroke_endpoint_ids(order[i - 1], flags[i - 1])
                cost += connector_cost(strokes, connector_model, prev_end_id, start_id)
            if i + 1 < n:
                next_start_id, _ = stroke_endpoint_ids(order[i + 1], flags[i + 1])
                cost += connector_cost(strokes, connector_model, end_id, next_start_id)
            return cost

        old_cost = local_cost(old_flag)
        new_cost = local_cost(not old_flag)
        if new_cost + 1e-6 < old_cost:
            flags[i] = not old_flag
            route.connector_length += new_cost - old_cost
            improved = True
        else:
            flags[i] = old_flag
    return improved


def two_opt_route(
    strokes: Sequence[Stroke],
    route: Route,
    seconds: float,
    connector_model: ConnectorModel | None = None,
    fixed_start: bool = False,
) -> Route:
    if len(route.order) < 4 or seconds <= 0:
        return route

    deadline = time.monotonic() + seconds
    order = route.order
    flags = route.reversed_flags
    n = len(order)
    improved = True

    while improved and time.monotonic() < deadline:
        improved = improve_orientations(strokes, route, connector_model, fixed_start=fixed_start)
        first_mutable = 1 if fixed_start else 0
        for start in range(first_mutable, n - 1):
            if time.monotonic() >= deadline:
                break
            start_start_id, start_end_id = stroke_endpoint_ids(order[start], flags[start])
            left_end_id = (
                stroke_endpoint_ids(order[start - 1], flags[start - 1])[1] if start > 0 else None
            )
            old_left = (
                connector_cost(strokes, connector_model, left_end_id, start_start_id)
                if left_end_id is not None
                else 0.0
            )
            for end in range(start + 1, n):
                end_start_id, end_end_id = stroke_endpoint_ids(order[end], flags[end])
                right_start_id = (
                    stroke_endpoint_ids(order[end + 1], flags[end + 1])[0]
                    if end + 1 < n
                    else None
                )
                old_cost = old_left
                new_cost = 0.0
                if left_end_id is not None:
                    new_cost += connector_cost(strokes, connector_model, left_end_id, end_end_id)
                if right_start_id is not None:
                    old_cost += connector_cost(strokes, connector_model, end_end_id, right_start_id)
                    new_cost += connector_cost(
                        strokes, connector_model, start_start_id, right_start_id
                    )
                if new_cost + 1e-6 < old_cost:
                    order[start : end + 1] = reversed(order[start : end + 1])
                    flags[start : end + 1] = [not flag for flag in reversed(flags[start : end + 1])]
                    route.connector_length += new_cost - old_cost
                    improved = True
                    break
            if improved:
                break

    route.connector_length = route_cost(
        strokes, route.order, route.reversed_flags, connector_model
    )
    return route


def closest_endpoint_seed(
    strokes: Sequence[Stroke], target: np.ndarray
) -> tuple[int, bool]:
    starts = np.asarray([stroke.start for stroke in strokes])
    ends = np.asarray([stroke.end for stroke in strokes])
    endpoints = np.stack([starts, ends], axis=1)
    flat = endpoints.reshape(-1, 2)
    flat_position = int(np.argmin(((flat - target[None, :]) ** 2).sum(axis=1)))
    stroke_index = flat_position // 2
    endpoint_index = flat_position % 2
    return stroke_index, endpoint_index == 1


def solve_route(
    strokes: Sequence[Stroke],
    two_opt_seconds: float,
    connector_model: ConnectorModel | None = None,
    start_mode: str = "best",
    start_point: np.ndarray | None = None,
) -> Route:
    if not strokes:
        return Route([], [], 0.0)

    if start_mode == "center":
        if start_point is None:
            min_x, min_y, max_x, max_y = bounds_from_strokes(strokes, (1, 1))
            start_point = np.asarray([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5])
        start_index, reversed_flag = closest_endpoint_seed(strokes, start_point)
        route = nearest_neighbor_route(strokes, start_index, reversed_flag, connector_model)
        return two_opt_route(
            strokes, route, two_opt_seconds, connector_model, fixed_start=True
        )

    best: Route | None = None
    for start_index, reversed_flag in endpoint_seed_indices(strokes):
        candidate = nearest_neighbor_route(strokes, start_index, reversed_flag, connector_model)
        if best is None or candidate.connector_length < best.connector_length:
            best = candidate
    assert best is not None
    return two_opt_route(strokes, best, two_opt_seconds, connector_model)


def oriented_points(stroke: Stroke, reversed_flag: bool) -> np.ndarray:
    return stroke.points[::-1] if reversed_flag else stroke.points


def continuous_points(
    strokes: Sequence[Stroke], route: Route, connector_model: ConnectorModel | None = None
) -> np.ndarray:
    chunks = continuous_runs(strokes, route, connector_model)
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def continuous_runs(
    strokes: Sequence[Stroke], route: Route, connector_model: ConnectorModel | None = None
) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    previous_end_id: int | None = None
    previous_end_point: np.ndarray | None = None
    for index, flag in zip(route.order, route.reversed_flags):
        start_id, end_id = stroke_endpoint_ids(index, flag)
        points = oriented_points(strokes[index], flag)
        if previous_end_id is not None:
            if connector_model is None:
                assert previous_end_point is not None
                connector = np.vstack([previous_end_point, points[0]])
            else:
                connector, _, _ = connector_model.connector_points_and_widths(
                    previous_end_id, start_id
                )
            chunks.append(connector)
        chunks.append(points)
        previous_end_id = end_id
        previous_end_point = points[-1]
    return chunks


def compact_polyline(points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    compacted = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - compacted[-1])) > tolerance:
            compacted.append(point)
    return np.asarray(compacted, dtype=np.float64)


def draw_fixed_opacity_polyline(
    base: Image.Image,
    points: np.ndarray,
    stroke_width: float,
    opacity: float,
    scale: int,
) -> None:
    if len(points) < 2:
        return
    scaled = np.asarray(points, dtype=np.float64) * scale
    xy = [tuple(point) for point in scaled]
    layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    width = max(1, int(round(stroke_width * scale)))
    alpha = int(round(np.clip(opacity, 0.0, 1.0) * 255))
    try:
        draw.line(xy, fill=(0, 0, 0, alpha), width=width, joint="curve")
    except TypeError:
        draw.line(xy, fill=(0, 0, 0, alpha), width=width)
    base.alpha_composite(layer)


def render_fixed_opacity_runs(
    runs: Sequence[np.ndarray],
    size: tuple[int, int],
    stroke_width: float = 10.0,
    opacity: float = 0.30,
    scale: int = 4,
) -> Image.Image:
    width, height = size
    canvas = Image.new("RGBA", (width * scale, height * scale), (255, 255, 255, 255))
    for run in runs:
        draw_fixed_opacity_polyline(canvas, run, stroke_width, opacity, scale)
    if scale != 1:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        canvas = canvas.resize((width, height), resample)
    return canvas.convert("RGB")


def route_connector_records(
    strokes: Sequence[Stroke], route: Route, connector_model: ConnectorModel | None
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    previous_end_id: int | None = None
    previous_stroke: int | None = None
    for index, flag in zip(route.order, route.reversed_flags):
        start_id, end_id = stroke_endpoint_ids(index, flag)
        if previous_end_id is not None and previous_stroke is not None:
            if connector_model is None:
                from_point = strokes[previous_stroke].start if previous_end_id % 2 == 0 else strokes[previous_stroke].end
                to_point = strokes[index].start if start_id % 2 == 0 else strokes[index].end
                straight_length = float(np.linalg.norm(from_point - to_point))
                graph_length = math.inf
                snapped = False
                effective_cost = straight_length
                draw_length = straight_length
            else:
                straight_length = connector_model.straight_length(previous_end_id, start_id)
                graph_length = float(connector_model.graph_length_matrix[previous_end_id, start_id])
                connector_points, _, snapped = connector_model.connector_points_and_widths(
                    previous_end_id, start_id
                )
                effective_cost = connector_model.cost(previous_end_id, start_id)
                draw_length = polyline_length(connector_points)
            records.append(
                {
                    "from_stroke": int(previous_stroke),
                    "to_stroke": int(index),
                    "from_endpoint": int(previous_end_id),
                    "to_endpoint": int(start_id),
                    "snapped": bool(snapped),
                    "straight_length": float(straight_length),
                    "graph_length": float(graph_length) if math.isfinite(graph_length) else None,
                    "draw_length": float(draw_length),
                    "effective_cost": float(effective_cost),
                }
            )
        previous_end_id = end_id
        previous_stroke = index
    return records


def connector_summary(records: Sequence[dict[str, object]]) -> dict[str, float | int]:
    snapped = [record for record in records if record["snapped"]]
    straight = [record for record in records if not record["snapped"]]
    return {
        "connector_count": len(records),
        "snapped_connector_count": len(snapped),
        "straight_connector_count": len(straight),
        "connector_draw_length": float(sum(float(record["draw_length"]) for record in records)),
        "snapped_connector_length": float(sum(float(record["draw_length"]) for record in snapped)),
        "straight_connector_length": float(sum(float(record["draw_length"]) for record in straight)),
        "straight_connector_max_length": float(
            max((float(record["draw_length"]) for record in straight), default=0.0)
        ),
    }


def stroke_to_json(stroke: Stroke) -> dict[str, object]:
    return {
        "width": stroke.median_width,
        "length": stroke.length,
        "points": np.round(stroke.points, 3).tolist(),
        "widths": np.round(stroke.widths, 3).tolist(),
    }


def process_image(
    image_path: Path,
    out_dir: Path,
    threshold: int | None,
    threshold_offset: int,
    simplify_epsilon: float,
    min_length: float,
    min_width: float,
    render_scale: int,
    two_opt_seconds: float,
    width_mode: str,
    smooth_passes: int,
    avoid_strength: float,
    retrace_weight: float,
    snap_max_ratio: float,
    snap_cost_ratio: float,
    route_mode: str,
    raster_step: int,
    raster_max_ratio: float,
    raster_min_line_fraction: float,
    raster_projection_distance: float,
    raster_min_projection_fraction: float,
    start_mode: str,
) -> dict[str, object]:
    started = time.monotonic()
    gray = load_grayscale(image_path)
    height, width = gray.shape
    ink, otsu, used_threshold = make_ink_mask(gray, threshold, threshold_offset)

    width_map_started = time.monotonic()
    if width_mode == "edt":
        width_map = distance_to_background(ink)
    else:
        width_map = erosion_distance_to_background(ink)
    width_map_seconds = time.monotonic() - width_map_started

    thinning_started = time.monotonic()
    skeleton, thinning_iterations = zhang_suen_thin(ink)
    thinning_seconds = time.monotonic() - thinning_started

    stroke_started = time.monotonic()
    strokes = build_strokes(
        skeleton,
        width_map,
        simplify_epsilon=simplify_epsilon,
        min_length=min_length,
        min_width=min_width,
        smooth_passes=smooth_passes,
    )
    stroke_seconds = time.monotonic() - stroke_started

    connector_started = time.monotonic()
    connector_model = build_connector_model(
        strokes,
        (width, height),
        ink=ink,
        avoid_strength=avoid_strength,
        retrace_weight=retrace_weight,
        snap_max_ratio=snap_max_ratio,
        snap_cost_ratio=snap_cost_ratio,
        raster_step=raster_step,
        raster_max_ratio=raster_max_ratio,
        raster_min_line_fraction=raster_min_line_fraction,
        raster_projection_distance=raster_projection_distance,
        raster_min_projection_fraction=raster_min_projection_fraction,
    )
    connector_seconds = time.monotonic() - connector_started

    route_started = time.monotonic()
    if route_mode == "component":
        route = solve_component_walk_route(strokes, connector_model=connector_model)
    else:
        route = solve_route(
            strokes,
            two_opt_seconds=two_opt_seconds,
            connector_model=connector_model,
            start_mode=start_mode,
            start_point=np.asarray([width * 0.5, height * 0.5], dtype=np.float64),
        )
    route_seconds = time.monotonic() - route_started

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    size = (width, height)

    connector_records = route_connector_records(strokes, route, connector_model)
    connectors = connector_summary(connector_records)
    routed_runs = continuous_runs(strokes, route, connector_model)
    routed_points = (
        np.concatenate(routed_runs, axis=0)
        if routed_runs
        else np.empty((0, 2), dtype=np.float64)
    )
    continuous_path_points = compact_polyline(routed_points)
    actual_start_point = (
        np.round(oriented_points(strokes[route.order[0]], route.reversed_flags[0])[0], 3).tolist()
        if route.order
        else None
    )
    requested_start_point = {
        "x": float(width * 0.5),
        "y": float(height * 0.5),
    }

    continuous_fixed_path = out_dir / f"{stem}_continuous_fixed_10px_30pct_nodots.png"
    json_path = out_dir / f"{stem}_vectors.json"

    render_fixed_opacity_runs(
        routed_runs, size, stroke_width=10.0, opacity=0.30, scale=render_scale
    ).save(
        continuous_fixed_path
    )

    route_indices = [
        {"stroke": int(index), "reversed": bool(flag)}
        for index, flag in zip(route.order, route.reversed_flags)
    ]
    total_stroke_length = float(sum(stroke.length for stroke in strokes))
    elapsed = time.monotonic() - started
    result: dict[str, object] = {
        "source": str(image_path),
        "width": width,
        "height": height,
        "otsu_threshold": otsu,
        "threshold": used_threshold,
        "width_mode": width_mode,
        "smooth_passes": smooth_passes,
        "avoid_strength": avoid_strength,
        "retrace_weight": retrace_weight,
        "snap_max_ratio": snap_max_ratio,
        "snap_cost_ratio": snap_cost_ratio,
        "route_mode": route_mode,
        "start_mode": start_mode,
        "start_point": requested_start_point,
        "actual_start_point": actual_start_point,
        "raster_step": raster_step,
        "raster_max_ratio": raster_max_ratio,
        "raster_min_line_fraction": raster_min_line_fraction,
        "raster_projection_distance": raster_projection_distance,
        "raster_min_projection_fraction": raster_min_projection_fraction,
        "ink_pixels": int(ink.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "thinning_iterations": thinning_iterations,
        "stroke_count": len(strokes),
        "point_count": int(sum(len(stroke.points) for stroke in strokes)),
        "total_stroke_length": total_stroke_length,
        "connector_length": route.connector_length,
        "connectors": connectors,
        "connector_to_stroke_ratio": float(route.connector_length / total_stroke_length)
        if total_stroke_length
        else 0.0,
        "continuous_path": {
            "type": "polyline",
            "closed": False,
            "point_count": int(len(continuous_path_points)),
            "raw_point_count": int(len(routed_points)),
            "length": polyline_length(continuous_path_points),
            "points": np.round(continuous_path_points, 3).tolist(),
        },
        "timings": {
            "width_map_seconds": width_map_seconds,
            "thinning_seconds": thinning_seconds,
            "stroke_seconds": stroke_seconds,
            "connector_model_seconds": connector_seconds,
            "route_seconds": route_seconds,
            "total_seconds": elapsed,
        },
        "outputs": {
            "vectors_json": str(json_path),
            "continuous_fixed_10px_30pct_nodots_png": str(continuous_fixed_path),
        },
        "route": route_indices,
        "connector_records": connector_records,
        "strokes": [stroke_to_json(stroke) for stroke in strokes],
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def collect_inputs(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("input-coloring")],
        help="Image files or directories to process.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output-vectors"),
        help="Directory for the vectors JSON and fixed-opacity PNG outputs.",
    )
    parser.add_argument("--threshold", type=int, default=None, help="Ink threshold. Defaults to Otsu plus offset.")
    parser.add_argument("--threshold-offset", type=int, default=85, help="Added to Otsu threshold when --threshold is omitted.")
    parser.add_argument("--simplify", type=float, default=0.7, help="Ramer-Douglas-Peucker tolerance in pixels.")
    parser.add_argument("--smooth-passes", type=int, default=0, help="Centerline smoothing passes before simplification.")
    parser.add_argument("--min-length", type=float, default=4.0, help="Drop skeleton fragments shorter than this many pixels.")
    parser.add_argument("--min-width", type=float, default=1.4, help="Minimum rendered vector stroke width.")
    parser.add_argument("--render-scale", type=int, default=4, help="Supersampling scale for the fixed-opacity PNG render.")
    parser.add_argument(
        "--width-mode",
        choices=("erosion", "edt"),
        default="erosion",
        help="Use fast erosion widths by default, or exact Euclidean distance widths.",
    )
    parser.add_argument(
        "--two-opt-seconds",
        type=float,
        default=1.5,
        help="Seconds spent improving each route after nearest-neighbor seeding.",
    )
    parser.add_argument(
        "--route-mode",
        choices=("component", "tsp"),
        default="tsp",
        help="component walks connected stroke graphs with backtracking; tsp visits each stroke once.",
    )
    parser.add_argument(
        "--start-mode",
        choices=("center", "best"),
        default="center",
        help="For tsp routes, start at the endpoint nearest the image center or the lowest-cost seed.",
    )
    parser.add_argument(
        "--avoid-strength",
        type=float,
        default=8.0,
        help="Penalty strength for straight connectors through the central face/neck mask.",
    )
    parser.add_argument(
        "--retrace-weight",
        type=float,
        default=0.45,
        help="Cost multiplier for connector paths that retrace existing vector strokes.",
    )
    parser.add_argument(
        "--snap-max-ratio",
        type=float,
        default=2.75,
        help="Use an existing-stroke connector only if it is at most this many times the straight distance.",
    )
    parser.add_argument(
        "--snap-cost-ratio",
        type=float,
        default=1.05,
        help="Use a snapped connector when its weighted cost is at most this ratio of straight weighted cost.",
    )
    parser.add_argument(
        "--raster-step",
        type=int,
        default=2,
        help="Pixel step for coarse A* connector routing over the ink/avoidance cost image.",
    )
    parser.add_argument(
        "--raster-max-ratio",
        type=float,
        default=2.75,
        help="A* connector paths are used only if their drawn length is within this straight-line ratio.",
    )
    parser.add_argument(
        "--raster-min-line-fraction",
        type=float,
        default=0.25,
        help="Reject A* connector paths unless this fraction of coarse path cells touch existing ink.",
    )
    parser.add_argument(
        "--raster-projection-distance",
        type=float,
        default=7.0,
        help="Project A* connector points onto extracted vector segments within this pixel distance.",
    )
    parser.add_argument(
        "--raster-min-projection-fraction",
        type=float,
        default=0.8,
        help="Reject A* connector paths unless this fraction of their length projects onto vector segments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = collect_inputs(args.inputs)
    if not image_paths:
        print("No input images found.")
        return 2

    for image_path in image_paths:
        print(f"Processing {image_path}...", flush=True)
        result = process_image(
            image_path=image_path,
            out_dir=args.out_dir,
            threshold=args.threshold,
            threshold_offset=args.threshold_offset,
            simplify_epsilon=args.simplify,
            min_length=args.min_length,
            min_width=args.min_width,
            render_scale=max(1, args.render_scale),
            two_opt_seconds=max(0.0, args.two_opt_seconds),
            width_mode=args.width_mode,
            smooth_passes=max(0, args.smooth_passes),
            avoid_strength=max(0.0, args.avoid_strength),
            retrace_weight=max(0.0, args.retrace_weight),
            snap_max_ratio=max(1.0, args.snap_max_ratio),
            snap_cost_ratio=max(0.0, args.snap_cost_ratio),
            route_mode=args.route_mode,
            raster_step=max(1, args.raster_step),
            raster_max_ratio=max(1.0, args.raster_max_ratio),
            raster_min_line_fraction=max(0.0, args.raster_min_line_fraction),
            raster_projection_distance=max(0.0, args.raster_projection_distance),
            raster_min_projection_fraction=max(0.0, args.raster_min_projection_fraction),
            start_mode=args.start_mode,
        )
        timings = result["timings"]
        print(
            "  strokes={stroke_count} points={point_count} connector_cost={connector_length:.1f} "
            "snapped={snapped}/{connectors} straight_max={straight_max:.1f} "
            "ratio={connector_to_stroke_ratio:.3f} time={time:.2f}s".format(
                stroke_count=result["stroke_count"],
                point_count=result["point_count"],
                connector_length=result["connector_length"],
                snapped=result["connectors"]["snapped_connector_count"],
                connectors=result["connectors"]["connector_count"],
                straight_max=result["connectors"]["straight_connector_max_length"],
                connector_to_stroke_ratio=result["connector_to_stroke_ratio"],
                time=timings["total_seconds"],
            )
            ,
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
