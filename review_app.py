#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DECISIONS = {"accepted", "passing", "rejected"}
EXCLUDED_IMAGE_NAME_TOKENS = ("contact-sheet",)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Outline Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f5f2;
      --panel: #ffffff;
      --ink: #1d1f21;
      --muted: #6b6f76;
      --line: #d7d7d2;
      --good: #2f7d4f;
      --pass: #b07a24;
      --bad: #b64242;
      --focus: #1f5f9d;
      --unknown: #8a8175;
    }

    * { box-sizing: border-box; }

    html, body {
      height: 100%;
      margin: 0;
      overflow: hidden;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }

    button {
      font: inherit;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      min-height: 36px;
      padding: 0 12px;
      border-radius: 6px;
      cursor: pointer;
    }

    button:hover { border-color: #a9aca6; }
    button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

    .shell {
      height: 100vh;
      height: 100dvh;
      min-height: 0;
      max-height: 100vh;
      max-height: 100dvh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      overflow: hidden;
    }

    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      min-height: 56px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(10px);
    }

    .titleline {
      min-width: 0;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 12px;
      align-items: baseline;
    }

    .index {
      font-weight: 700;
      font-size: 15px;
      white-space: nowrap;
    }

    .filename {
      color: var(--muted);
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .stats {
      display: flex;
      gap: 8px;
      align-items: center;
      white-space: nowrap;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      font-size: 12px;
      font-weight: 650;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--unknown);
    }

    .pill.accepted .dot { background: var(--good); }
    .pill.passing .dot { background: var(--pass); }
    .pill.rejected .dot { background: var(--bad); }
    .pill.undecided .dot { background: var(--unknown); }

    .main {
      min-height: 0;
      height: 100%;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
    }

    .stage {
      min-height: 0;
      min-width: 0;
      overflow: hidden;
      display: grid;
      place-items: center;
      padding: 16px;
      background:
        linear-gradient(45deg, rgba(0,0,0,0.035) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(0,0,0,0.035) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(0,0,0,0.035) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(0,0,0,0.035) 75%);
      background-size: 28px 28px;
      background-position: 0 0, 0 14px, 14px -14px, -14px 0;
    }

    .imagewrap {
      width: min(100%, calc(100vh - 88px));
      width: min(100%, calc(100dvh - 88px));
      height: min(100%, calc(100vh - 88px));
      height: min(100%, calc(100dvh - 88px));
      max-width: 1160px;
      max-height: 100%;
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
    }

    .review-image {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: auto;
      background: #fff;
      border: 1px solid var(--line);
      box-shadow: 0 12px 38px rgba(22, 24, 28, 0.18);
    }

    .sidebar {
      min-height: 0;
      height: 100%;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      border-left: 1px solid var(--line);
      background: #fbfbf8;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfbf8;
      z-index: 1;
    }

    .actions button {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 650;
    }

    .actions .accept { border-color: rgba(47, 125, 79, 0.4); color: #245c3c; }
    .actions .passing { border-color: rgba(176, 122, 36, 0.45); color: #7a5418; }
    .actions .reject { border-color: rgba(182, 66, 66, 0.4); color: #8c3030; }
    .actions .clear { color: #555a60; }
    .actions .filter.active {
      border-color: rgba(31, 95, 157, 0.55);
      background: #edf4fb;
      color: #164f84;
    }

    .strip {
      min-height: 0;
      height: 100%;
      overflow-x: hidden;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      grid-auto-rows: 84px;
      gap: 8px;
      align-content: start;
    }

    .thumb {
      position: relative;
      border: 2px solid transparent;
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
      padding: 0;
      min-height: 0;
    }

    .thumb img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .thumb.current { border-color: var(--focus); }
    .thumb.accepted::after,
    .thumb.passing::after,
    .thumb.rejected::after,
    .thumb.undecided::after {
      content: "";
      position: absolute;
      top: 5px;
      right: 5px;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 2px solid #fff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .thumb.accepted::after { background: var(--good); }
    .thumb.passing::after { background: var(--pass); }
    .thumb.rejected::after { background: var(--bad); }
    .thumb.undecided::after { background: var(--unknown); }

    .empty {
      padding: 32px;
      color: var(--muted);
      text-align: center;
    }

    @media (max-width: 900px) {
      .main {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(0, 1fr) auto;
      }
      .sidebar {
        border-left: 0;
        border-top: 1px solid var(--line);
        grid-template-rows: auto 126px;
        height: 100%;
      }
      .actions {
        grid-template-columns: repeat(5, 1fr);
      }
      .strip {
        display: flex;
        overflow-x: auto;
        overflow-y: hidden;
      }
      .thumb {
        flex: 0 0 82px;
        height: 82px;
      }
      .topbar {
        grid-template-columns: 1fr;
        gap: 8px;
      }
      .stats {
        overflow-x: auto;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="titleline">
        <div id="index" class="index">0 / 0</div>
        <div id="filename" class="filename"></div>
      </div>
      <div class="stats">
        <span id="status" class="pill undecided"><span class="dot"></span><span>Undecided</span></span>
        <span id="accepted" class="pill accepted"><span class="dot"></span><span>0</span></span>
        <span id="passing" class="pill passing"><span class="dot"></span><span>0</span></span>
        <span id="rejected" class="pill rejected"><span class="dot"></span><span>0</span></span>
        <span id="undecided" class="pill undecided"><span class="dot"></span><span>0</span></span>
      </div>
    </header>
    <main class="main">
      <section class="stage">
        <div id="imagewrap" class="imagewrap"></div>
      </section>
      <aside class="sidebar">
        <div class="actions">
          <button class="accept" id="accept">Accept</button>
          <button class="passing" id="passingButton">Passing</button>
          <button class="reject" id="reject">Reject</button>
          <button class="clear" id="clear">Clear</button>
          <button class="filter" id="unmarkedOnly">Unmarked only</button>
        </div>
        <div id="strip" class="strip"></div>
      </aside>
    </main>
  </div>

  <script>
    const state = {
      allImages: [],
      images: [],
      decisions: new Map(),
      index: 0,
      filterUnmarked: false,
      thumbObserver: null,
    };

    const els = {
      index: document.getElementById("index"),
      filename: document.getElementById("filename"),
      status: document.getElementById("status"),
      accepted: document.getElementById("accepted"),
      passing: document.getElementById("passing"),
      rejected: document.getElementById("rejected"),
      undecided: document.getElementById("undecided"),
      imagewrap: document.getElementById("imagewrap"),
      strip: document.getElementById("strip"),
      accept: document.getElementById("accept"),
      passingButton: document.getElementById("passingButton"),
      reject: document.getElementById("reject"),
      clear: document.getElementById("clear"),
      unmarkedOnly: document.getElementById("unmarkedOnly"),
    };

    function decisionFor(id) {
      return state.decisions.get(id) || null;
    }

    function decisionClass(decision) {
      return decision || "undecided";
    }

    function counts() {
      let accepted = 0;
      let passing = 0;
      let rejected = 0;
      for (const value of state.decisions.values()) {
        if (value === "accepted") accepted++;
        if (value === "passing") passing++;
        if (value === "rejected") rejected++;
      }
      return {
        accepted,
        passing,
        rejected,
        undecided: state.allImages.length - accepted - passing - rejected,
      };
    }

    function clampIndex(index) {
      if (!state.images.length) return 0;
      return Math.max(0, Math.min(state.images.length - 1, index));
    }

    function currentImage() {
      return state.images[state.index] || null;
    }

    function setStatusPill(decision) {
      const label = decision === "accepted" ? "Accepted" :
        decision === "passing" ? "Passing" :
        decision === "rejected" ? "Rejected" : "Undecided";
      els.status.className = `pill ${decisionClass(decision)}`;
      els.status.innerHTML = `<span class="dot"></span><span>${label}</span>`;
    }

    function updateStats() {
      const next = counts();
      els.accepted.querySelector("span:last-child").textContent = next.accepted;
      els.passing.querySelector("span:last-child").textContent = next.passing;
      els.rejected.querySelector("span:last-child").textContent = next.rejected;
      els.undecided.querySelector("span:last-child").textContent = next.undecided;
      els.unmarkedOnly.classList.toggle("active", state.filterUnmarked);
    }

    function renderMain() {
      const image = currentImage();
      if (!image) {
        els.index.textContent = "0 / 0";
        els.filename.textContent = "";
        els.imagewrap.innerHTML = `<div class="empty">${state.filterUnmarked ? "No unmarked images" : "No images found"}</div>`;
        setStatusPill(null);
        updateStats();
        return;
      }

      const decision = decisionFor(image.id);
      els.index.textContent = `${state.index + 1} / ${state.images.length}`;
      els.filename.textContent = image.filename;
      els.imagewrap.innerHTML = `<img class="review-image" src="${image.url}" alt="">`;
      setStatusPill(decision);
      updateStats();
    }

    function renderThumbs() {
      els.strip.innerHTML = "";
      const fragment = document.createDocumentFragment();
      for (const [index, image] of state.images.entries()) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `thumb ${decisionClass(decisionFor(image.id))}`;
        button.dataset.index = String(index);
        button.title = image.filename;
        button.innerHTML = `<img loading="lazy" src="${image.url}" alt="">`;
        button.addEventListener("click", () => go(index));
        fragment.append(button);
      }
      els.strip.append(fragment);
      syncThumbs();
    }

    function syncThumbs() {
      const thumbs = els.strip.querySelectorAll(".thumb");
      thumbs.forEach((thumb, index) => {
        const image = state.images[index];
        thumb.className = `thumb ${decisionClass(decisionFor(image.id))}`;
        if (index === state.index) {
          thumb.classList.add("current");
        }
      });
      const current = els.strip.querySelector(".thumb.current");
      if (current) {
        current.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    }

    function applyFilter(preferredId) {
      state.images = state.filterUnmarked ?
        state.allImages.filter((image) => !state.decisions.has(image.id)) :
        [...state.allImages];

      const nextIndex = preferredId ?
        state.images.findIndex((image) => image.id === preferredId) :
        -1;
      state.index = nextIndex >= 0 ? nextIndex : clampIndex(state.index);
    }

    function render() {
      renderMain();
      syncThumbs();
    }

    function go(index) {
      state.index = clampIndex(index);
      render();
    }

    function next() {
      go(state.index + 1);
    }

    function previous() {
      go(state.index - 1);
    }

    async function saveDecision(decision, advance) {
      const image = currentImage();
      if (!image) return;
      const oldIndex = state.index;

      const response = await fetch("/api/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: image.id, decision }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      if (decision === null) {
        state.decisions.delete(image.id);
      } else {
        state.decisions.set(image.id, decision);
      }

      if (state.filterUnmarked && decision !== null) {
        applyFilter(null);
        state.index = clampIndex(Math.min(oldIndex, state.images.length - 1));
        renderThumbs();
        renderMain();
      } else if (advance) {
        applyFilter(image.id);
        state.index = clampIndex(oldIndex + 1);
        renderThumbs();
        renderMain();
      } else {
        applyFilter(image.id);
        renderThumbs();
        renderMain();
      }
    }

    function toggleUnmarkedOnly() {
      const image = currentImage();
      state.filterUnmarked = !state.filterUnmarked;
      applyFilter(image ? image.id : null);
      renderThumbs();
      renderMain();
    }

    async function init() {
      const response = await fetch("/api/state");
      if (!response.ok) {
        els.imagewrap.innerHTML = `<div class="empty">${await response.text()}</div>`;
        return;
      }
      const data = await response.json();
      state.allImages = data.images;
      state.decisions = new Map(Object.entries(data.decisions));
      applyFilter(null);
      const firstUndecided = state.images.findIndex((image) => !state.decisions.has(image.id));
      state.index = firstUndecided >= 0 ? firstUndecided : 0;
      renderThumbs();
      renderMain();
    }

    els.accept.addEventListener("click", () => saveDecision("accepted", true));
    els.passingButton.addEventListener("click", () => saveDecision("passing", true));
    els.reject.addEventListener("click", () => saveDecision("rejected", true));
    els.clear.addEventListener("click", () => saveDecision(null, false));
    els.unmarkedOnly.addEventListener("click", toggleUnmarkedOnly);

    window.addEventListener("keydown", (event) => {
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (event.code === "Space") {
        event.preventDefault();
        saveDecision("accepted", true);
      } else if (event.code === "KeyP") {
        event.preventDefault();
        saveDecision("passing", true);
      } else if (event.code === "Tab") {
        event.preventDefault();
        saveDecision("rejected", true);
      } else if (event.code === "ArrowRight") {
        event.preventDefault();
        next();
      } else if (event.code === "ArrowLeft") {
        event.preventDefault();
        previous();
      } else if (event.code === "Delete") {
        event.preventDefault();
        saveDecision(null, false);
      }
    });

    init();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review aligned outline overlays.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--images",
        action="append",
        help="Image directory to review. Can be repeated.",
    )
    parser.add_argument(
        "--image-glob",
        action="append",
        help="Image glob to review. Can be repeated, for example 'faces/hk__*.jpg'.",
    )
    parser.add_argument("--db", default="review_decisions.sqlite3")
    return parser.parse_args()


def image_files_from_dir(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_files_from_glob(pattern: str) -> list[Path]:
    return sorted(
        Path(path)
        for path in glob.glob(pattern)
        if Path(path).is_file() and Path(path).suffix.lower() in IMAGE_EXTENSIONS
    )


def image_records(image_dirs: list[Path], image_globs: list[str]) -> list[ImageRecord]:
    files: list[Path] = []
    for images_dir in image_dirs:
        files.extend(image_files_from_dir(images_dir))
    for pattern in image_globs:
        files.extend(image_files_from_glob(pattern))

    records: list[ImageRecord] = []
    seen: set[str] = set()
    for path in sorted(files, key=lambda item: item.name):
        image_id = path.name
        if any(token in image_id.lower() for token in EXCLUDED_IMAGE_NAME_TOKENS):
            continue
        if image_id in seen:
            continue
        seen.add(image_id)
        records.append(ImageRecord(image_id=image_id, path=path.resolve()))
    return records


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'decisions'"
        ).fetchone()
        if schema is not None and "passing" not in (schema[0] or ""):
            connection.execute(
                """
                CREATE TABLE decisions_next (
                    image_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL CHECK (decision IN ('accepted', 'passing', 'rejected')),
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO decisions_next (image_id, decision, updated_at)
                SELECT image_id, decision, updated_at
                FROM decisions
                WHERE decision IN ('accepted', 'rejected')
                """
            )
            connection.execute("DROP TABLE decisions")
            connection.execute("ALTER TABLE decisions_next RENAME TO decisions")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                image_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL CHECK (decision IN ('accepted', 'passing', 'rejected')),
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def load_decisions(db_path: Path, valid_ids: set[str]) -> dict[str, str]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT image_id, decision FROM decisions ORDER BY image_id"
        ).fetchall()
    return {
        image_id: decision
        for image_id, decision in rows
        if image_id in valid_ids and decision in DECISIONS
    }


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/state":
            self.handle_state()
            return

        if parsed.path.startswith("/images/"):
            self.handle_image(parsed.path.removeprefix("/images/"))
            return

        self.send_text("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/decision":
            self.handle_decision()
            return
        self.send_text("Not found", HTTPStatus.NOT_FOUND)

    def read_json_body(self) -> dict[str, object] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def handle_state(self) -> None:
        records = self.server.image_records()
        valid_ids = {record.image_id for record in records}
        decisions = load_decisions(self.server.db_path, valid_ids)
        images = [
            {
                "id": record.image_id,
                "filename": record.image_id,
                "url": f"/images/{quote(record.image_id)}",
            }
            for record in records
        ]
        self.send_json({"images": images, "decisions": decisions})

    def handle_image(self, encoded_name: str) -> None:
        name = unquote(encoded_name)
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            self.send_text("Invalid image path", HTTPStatus.BAD_REQUEST)
            return

        image_map = self.server.image_map()
        resolved = image_map.get(name)
        if resolved is None:
            self.send_text("Image not found", HTTPStatus.NOT_FOUND)
            return

        mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(data)

    def handle_decision(self) -> None:
        payload = self.read_json_body()
        if payload is None:
            self.send_text("Invalid JSON", HTTPStatus.BAD_REQUEST)
            return

        image_id = payload.get("id")
        decision = payload.get("decision")
        if not isinstance(image_id, str) or not image_id:
            self.send_text("Invalid image id", HTTPStatus.BAD_REQUEST)
            return

        records = self.server.image_records()
        valid_ids = {record.image_id for record in records}
        if image_id not in valid_ids:
            self.send_text("Image id not found", HTTPStatus.NOT_FOUND)
            return

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(self.server.db_path) as connection:
            if decision is None:
                connection.execute("DELETE FROM decisions WHERE image_id = ?", (image_id,))
            elif decision in DECISIONS:
                connection.execute(
                    """
                    INSERT INTO decisions (image_id, decision, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        decision = excluded.decision,
                        updated_at = excluded.updated_at
                    """,
                    (image_id, decision, now),
                )
            else:
                self.send_text("Invalid decision", HTTPStatus.BAD_REQUEST)
                return
            connection.commit()

        decisions = load_decisions(self.server.db_path, valid_ids)
        self.send_json(
            {
                "id": image_id,
                "decision": decision,
                "counts": {
                    "accepted": sum(1 for value in decisions.values() if value == "accepted"),
                    "passing": sum(1 for value in decisions.values() if value == "passing"),
                    "rejected": sum(1 for value in decisions.values() if value == "rejected"),
                    "undecided": len(valid_ids) - len(decisions),
                },
            }
        )


class ReviewServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        image_dirs: list[Path],
        image_globs: list[str],
        db_path: Path,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.image_dirs = [path.resolve() for path in image_dirs]
        self.image_globs = image_globs
        self.db_path = db_path.resolve()

    def image_records(self) -> list[ImageRecord]:
        return image_records(self.image_dirs, self.image_globs)

    def image_map(self) -> dict[str, Path]:
        return {record.image_id: record.path for record in self.image_records()}


def main() -> int:
    args = parse_args()
    image_dirs = [Path(path) for path in (args.images or [])]
    image_globs = args.image_glob or []
    if not image_dirs and not image_globs:
        image_dirs = [Path("overlaid-aligned")]
    db_path = Path(args.db)
    init_db(db_path)

    server = ReviewServer(
        (args.host, args.port),
        ReviewHandler,
        image_dirs,
        image_globs,
        db_path,
    )
    host, port = server.server_address
    print(f"Review app: http://{host}:{port}")
    for image_dir in image_dirs:
        print(f"Images: {image_dir.resolve()}")
    for image_glob in image_globs:
        print(f"Image glob: {image_glob}")
    print(f"Database: {db_path.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review app.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
