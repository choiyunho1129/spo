#!/usr/bin/env python3
"""Serve a lightweight web viewer for CRRL rollout/validation JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOT = REPO_ROOT / "crrl_verl_pr"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRRL Rollout Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #676b73;
      --line: #d9d7cf;
      --accent: #117a65;
      --accent-soft: #e3f2ed;
      --warn: #a6422b;
      --warn-soft: #f8e7df;
      --focus: #1c5fb8;
      --code: #14161a;
      --code-bg: #f1f2f2;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }

    header {
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fbfbf9;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .path {
      margin-top: 4px;
      max-width: 72vw;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .shell {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: calc(100vh - 73px);
    }

    aside {
      border-right: 1px solid var(--line);
      background: #efeee8;
      min-height: 100%;
      overflow: auto;
    }

    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 136px 104px auto auto;
      gap: 10px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    input,
    select,
    button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      letter-spacing: 0;
    }

    input,
    select {
      padding: 0 10px;
      min-width: 0;
    }

    button {
      padding: 0 12px;
      cursor: pointer;
      font-weight: 600;
    }

    button:hover {
      border-color: #a9a69c;
      background: #f6f6f2;
    }

    button:focus-visible,
    input:focus-visible,
    select:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 1px;
    }

    .files-title,
    .summary {
      padding: 14px 16px 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .file-list {
      display: grid;
      gap: 8px;
      padding: 0 12px 16px;
    }

    .file-button {
      width: 100%;
      height: auto;
      min-height: 70px;
      display: grid;
      gap: 4px;
      padding: 10px 12px;
      text-align: left;
      border-radius: 8px;
      background: #fbfbf8;
    }

    .file-button.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }

    .file-name {
      font-weight: 750;
      overflow-wrap: anywhere;
    }

    .file-meta {
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }

    .content {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(280px, 38%) minmax(0, 1fr);
    }

    .records {
      min-height: 0;
      overflow: auto;
      border-right: 1px solid var(--line);
      background: #fbfbf8;
    }

    .record-list {
      display: grid;
      gap: 8px;
      padding: 0 12px 16px;
    }

    .record-button {
      width: 100%;
      height: auto;
      min-height: 82px;
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      text-align: left;
      border-radius: 8px;
      background: var(--panel);
    }

    .record-button.active {
      border-color: var(--focus);
      background: #eef4fd;
    }

    .row {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }

    .line-no {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: nowrap;
    }

    .badge {
      min-width: 52px;
      padding: 3px 7px;
      border-radius: 999px;
      text-align: center;
      font-size: 12px;
      font-weight: 800;
      color: var(--accent);
      background: var(--accent-soft);
    }

    .badge.bad {
      color: var(--warn);
      background: var(--warn-soft);
    }

    .snippet {
      color: var(--muted);
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      overflow-wrap: anywhere;
    }

    .kv {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .detail {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      padding: 18px;
      background: var(--panel);
    }

    .detail-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }

    .detail-title {
      min-width: 0;
    }

    .detail-title strong {
      display: block;
      font-size: 18px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .detail-title span {
      color: var(--muted);
      font-size: 12px;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 10px 0 18px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfbf8;
      min-width: 0;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 700;
    }

    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 15px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .section {
      margin-top: 16px;
    }

    .section h2 {
      margin: 0 0 8px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }

    pre {
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--code-bg);
      color: var(--code);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }

    .empty,
    .error {
      padding: 18px;
      color: var(--muted);
    }

    .error {
      color: var(--warn);
    }

    @media (max-width: 980px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      .path {
        max-width: calc(100vw - 48px);
      }

      .shell,
      .content {
        grid-template-columns: 1fr;
      }

      aside {
        max-height: 260px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .toolbar {
        grid-template-columns: 1fr 1fr;
      }

      .records {
        max-height: 360px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CRRL Rollout Viewer</h1>
      <div class="path" id="dataPath"></div>
    </div>
    <button id="refreshButton">Refresh</button>
  </header>
  <div class="shell">
    <aside>
      <div class="files-title">Files</div>
      <div class="file-list" id="fileList"></div>
    </aside>
    <main>
      <div class="toolbar">
        <input id="searchInput" type="search" placeholder="Search input, output, pred, gts">
        <select id="scoreFilter">
          <option value="all">All scores</option>
          <option value="correct">Correct</option>
          <option value="wrong">Wrong</option>
          <option value="nonzero">Non-zero</option>
          <option value="zero">Zero</option>
        </select>
        <select id="limitSelect">
          <option>25</option>
          <option selected>50</option>
          <option>100</option>
          <option>200</option>
        </select>
        <button id="prevButton">Prev</button>
        <button id="nextButton">Next</button>
      </div>
      <div class="content">
        <section class="records">
          <div class="summary" id="summary"></div>
          <div class="record-list" id="recordList"></div>
        </section>
        <section class="detail" id="detail"></section>
      </div>
    </main>
  </div>

  <script>
    const state = {
      files: [],
      selectedFile: null,
      records: [],
      selectedLine: null,
      total: 0,
      stats: null,
      offset: 0,
      error: null,
    };

    const el = {
      dataPath: document.getElementById("dataPath"),
      fileList: document.getElementById("fileList"),
      recordList: document.getElementById("recordList"),
      detail: document.getElementById("detail"),
      summary: document.getElementById("summary"),
      searchInput: document.getElementById("searchInput"),
      scoreFilter: document.getElementById("scoreFilter"),
      limitSelect: document.getElementById("limitSelect"),
      prevButton: document.getElementById("prevButton"),
      nextButton: document.getElementById("nextButton"),
      refreshButton: document.getElementById("refreshButton"),
    };

    function fmtBytes(bytes) {
      if (!Number.isFinite(bytes)) return "";
      const units = ["B", "KB", "MB", "GB"];
      let value = bytes;
      let idx = 0;
      while (value >= 1024 && idx < units.length - 1) {
        value /= 1024;
        idx += 1;
      }
      return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
    }

    function scoreOf(record) {
      const raw = record.score ?? record.reward;
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    }

    function shortText(value, maxLen = 180) {
      const text = String(value ?? "").replace(/\s+/g, " ").trim();
      if (text.length <= maxLen) return text;
      return text.slice(0, maxLen - 1) + "...";
    }

    function clear(node) {
      while (node.firstChild) node.removeChild(node.firstChild);
    }

    function node(tag, className, text) {
      const out = document.createElement(tag);
      if (className) out.className = className;
      if (text !== undefined) out.textContent = text;
      return out;
    }

    async function getJson(url) {
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok || payload.error) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    async function loadFiles() {
      state.error = null;
      try {
        const payload = await getJson("/api/files");
        state.files = payload.files;
        el.dataPath.textContent = payload.data_dir;
        if (!state.selectedFile && state.files.length) {
          state.selectedFile = state.files[state.files.length - 1].name;
        }
        if (state.files.length && !state.files.some((f) => f.name === state.selectedFile)) {
          state.selectedFile = state.files[state.files.length - 1].name;
        }
        renderFiles();
        await loadRecords(true);
      } catch (err) {
        state.error = err.message;
        renderFiles();
        renderRecords();
        renderDetail(null);
      }
    }

    function renderFiles() {
      clear(el.fileList);
      if (state.error) {
        el.fileList.appendChild(node("div", "error", state.error));
        return;
      }
      if (!state.files.length) {
        el.fileList.appendChild(node("div", "empty", "No JSONL files found."));
        return;
      }
      for (const file of state.files) {
        const button = node("button", "file-button");
        if (file.name === state.selectedFile) button.classList.add("active");
        button.addEventListener("click", async () => {
          state.selectedFile = file.name;
          state.offset = 0;
          state.selectedLine = null;
          renderFiles();
          await loadRecords(true);
        });

        button.appendChild(node("div", "file-name", file.name));
        button.appendChild(node("div", "file-meta", `${file.records} rows - ${fmtBytes(file.bytes)}`));
        button.appendChild(node("div", "file-meta", file.modified));
        el.fileList.appendChild(button);
      }
    }

    async function loadRecords(resetSelection = false) {
      if (!state.selectedFile) {
        state.records = [];
        state.total = 0;
        state.stats = null;
        renderRecords();
        renderDetail(null);
        return;
      }
      const params = new URLSearchParams({
        file: state.selectedFile,
        q: el.searchInput.value,
        score: el.scoreFilter.value,
        limit: el.limitSelect.value,
        offset: String(state.offset),
      });
      try {
        const payload = await getJson(`/api/records?${params.toString()}`);
        state.records = payload.records;
        state.total = payload.total;
        state.stats = payload.stats;
        if (resetSelection || !state.records.some((r) => r._line === state.selectedLine)) {
          state.selectedLine = state.records.length ? state.records[0]._line : null;
        }
        renderRecords();
        renderDetail(state.records.find((r) => r._line === state.selectedLine) || null);
      } catch (err) {
        state.records = [];
        state.total = 0;
        state.stats = null;
        state.error = err.message;
        renderRecords();
        renderDetail(null);
      }
    }

    function renderRecords() {
      clear(el.recordList);
      const limit = Number(el.limitSelect.value);
      const end = Math.min(state.offset + state.records.length, state.total);
      const stats = state.stats || {correct: 0, avg_score: null};
      const avg = stats.avg_score === null ? "n/a" : Number(stats.avg_score).toFixed(3);
      el.summary.textContent = state.selectedFile
        ? `${state.offset + (state.records.length ? 1 : 0)}-${end} / ${state.total} | correct ${stats.correct} | avg ${avg}`
        : "";
      el.prevButton.disabled = state.offset <= 0;
      el.nextButton.disabled = state.offset + limit >= state.total;

      if (state.error) {
        el.recordList.appendChild(node("div", "error", state.error));
        return;
      }
      if (!state.records.length) {
        el.recordList.appendChild(node("div", "empty", "No records match."));
        return;
      }

      for (const record of state.records) {
        const score = scoreOf(record);
        const button = node("button", "record-button");
        if (record._line === state.selectedLine) button.classList.add("active");
        button.addEventListener("click", () => {
          state.selectedLine = record._line;
          renderRecords();
          renderDetail(record);
        });

        const top = node("div", "row");
        top.appendChild(node("span", "line-no", `#${record._line}`));
        const badge = node("span", "badge", score === null ? "n/a" : score.toFixed(3));
        if (score !== null && score <= 0) badge.classList.add("bad");
        top.appendChild(badge);
        top.appendChild(node("span", "kv", `pred ${shortText(record.pred, 32)} / gts ${shortText(record.gts, 32)}`));
        button.appendChild(top);
        button.appendChild(node("div", "snippet", shortText(record.output || record.input, 220)));
        el.recordList.appendChild(button);
      }
    }

    function metric(label, value) {
      const box = node("div", "metric");
      box.appendChild(node("span", null, label));
      box.appendChild(node("strong", null, value));
      return box;
    }

    function preSection(label, value) {
      const section = node("div", "section");
      section.appendChild(node("h2", null, label));
      section.appendChild(node("pre", null, String(value ?? "")));
      return section;
    }

    function renderDetail(record) {
      clear(el.detail);
      if (!record) {
        el.detail.appendChild(node("div", "empty", "Select a record."));
        return;
      }

      const top = node("div", "detail-top");
      const title = node("div", "detail-title");
      title.appendChild(node("strong", null, `${state.selectedFile} #${record._line}`));
      title.appendChild(node("span", null, `step ${record.step ?? "n/a"}`));
      top.appendChild(title);

      const copyButton = node("button", null, "Copy output");
      copyButton.addEventListener("click", async () => {
        await navigator.clipboard.writeText(String(record.output ?? ""));
        copyButton.textContent = "Copied";
        setTimeout(() => { copyButton.textContent = "Copy output"; }, 900);
      });
      top.appendChild(copyButton);
      el.detail.appendChild(top);

      const grid = node("div", "metric-grid");
      const score = scoreOf(record);
      grid.appendChild(metric("Score", score === null ? "n/a" : score.toFixed(4)));
      grid.appendChild(metric("Reward", record.reward ?? "n/a"));
      grid.appendChild(metric("Pred", record.pred ?? "n/a"));
      grid.appendChild(metric("GTS", record.gts ?? "n/a"));
      el.detail.appendChild(grid);

      el.detail.appendChild(preSection("Input", record.input));
      el.detail.appendChild(preSection("Output", record.output));

      const extra = {...record};
      for (const key of ["_line", "input", "output", "gts", "score", "step", "reward", "pred"]) {
        delete extra[key];
      }
      if (Object.keys(extra).length) {
        el.detail.appendChild(preSection("Extra", JSON.stringify(extra, null, 2)));
      }
    }

    function debounce(fn, wait) {
      let timer = null;
      return (...args) => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => fn(...args), wait);
      };
    }

    el.searchInput.addEventListener("input", debounce(() => {
      state.offset = 0;
      state.selectedLine = null;
      loadRecords(true);
    }, 200));
    el.scoreFilter.addEventListener("change", () => {
      state.offset = 0;
      state.selectedLine = null;
      loadRecords(true);
    });
    el.limitSelect.addEventListener("change", () => {
      state.offset = 0;
      loadRecords(true);
    });
    el.prevButton.addEventListener("click", () => {
      const limit = Number(el.limitSelect.value);
      state.offset = Math.max(0, state.offset - limit);
      loadRecords(true);
    });
    el.nextButton.addEventListener("click", () => {
      const limit = Number(el.limitSelect.value);
      state.offset += limit;
      loadRecords(true);
    });
    el.refreshButton.addEventListener("click", loadFiles);

    loadFiles();
  </script>
</body>
</html>
"""


def find_default_data_dir() -> Path | None:
    if not DEFAULT_SEARCH_ROOT.exists():
        return None

    candidates = []
    for pattern in ("*/validation_data", "*/rollout_data"):
        for path in DEFAULT_SEARCH_ROOT.glob(pattern):
            if path.is_dir():
                files = list(path.glob("*.jsonl"))
                if files:
                    newest = max(file.stat().st_mtime for file in files)
                    candidates.append((newest, path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", help="Directory containing rollout/validation JSONL files.")
    parser.add_argument("--data-dir", dest="data_dir_option", help="Same as positional data_dir.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for remote access.")
    parser.add_argument("--port", type=int, default=8008, help="Bind port.")
    return parser.parse_args()


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def error_response(handler: BaseHTTPRequestHandler, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
    json_response(handler, {"error": message}, status)


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def file_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (sys.maxsize, path.name)


def safe_jsonl_path(data_dir: Path, name: str) -> Path:
    basename = os.path.basename(name)
    if basename != name or not basename.endswith(".jsonl"):
        raise ValueError("Invalid JSONL filename.")
    path = (data_dir / basename).resolve()
    if path.parent != data_dir.resolve() or not path.exists():
        raise ValueError("JSONL file does not exist.")
    return path


def to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def score_value(record: dict) -> float | None:
    score = to_float(record.get("score"))
    if score is not None:
        return score
    return to_float(record.get("reward"))


def matches_score_filter(record: dict, score_filter: str) -> bool:
    score = score_value(record)
    if score_filter == "all":
        return True
    if score_filter == "correct":
        return score is not None and score > 0
    if score_filter == "wrong":
        return score is not None and score <= 0
    if score_filter == "nonzero":
        return score is not None and score != 0
    if score_filter == "zero":
        return score is not None and score == 0
    return True


def matches_query(record: dict, query: str) -> bool:
    if not query:
        return True
    haystack_parts = []
    for key in ("input", "output", "gts", "pred", "score", "reward", "step"):
        value = record.get(key)
        if value is not None:
            haystack_parts.append(str(value))
    return query.casefold() in "\n".join(haystack_parts).casefold()


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                record = {"_parse_error": str(exc), "raw": line}
            if not isinstance(record, dict):
                record = {"value": record}
            record["_line"] = line_no
            yield record


class RolloutViewerHandler(BaseHTTPRequestHandler):
    server_version = "CRRLRolloutViewer/1.0"

    @property
    def data_dir(self) -> Path:
        return self.server.data_dir

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_index()
            return
        if parsed.path == "/api/files":
            self.serve_files()
            return
        if parsed.path == "/api/records":
            self.serve_records(parse_qs(parsed.query))
            return
        error_response(self, "Not found.", HTTPStatus.NOT_FOUND)

    def serve_index(self) -> None:
        data = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_files(self) -> None:
        files = []
        for path in sorted(self.data_dir.glob("*.jsonl"), key=file_sort_key):
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "records": count_lines(path),
                    "bytes": stat.st_size,
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                }
            )
        json_response(self, {"data_dir": str(self.data_dir), "files": files})

    def serve_records(self, query: dict[str, list[str]]) -> None:
        try:
            filename = query.get("file", [""])[0]
            path = safe_jsonl_path(self.data_dir, filename)
            search = query.get("q", [""])[0].strip()
            score_filter = query.get("score", ["all"])[0]
            limit = max(1, min(200, int(query.get("limit", ["50"])[0])))
            offset = max(0, int(query.get("offset", ["0"])[0]))
        except (TypeError, ValueError) as exc:
            error_response(self, str(exc))
            return

        records = []
        total = 0
        correct = 0
        score_sum = 0.0
        score_count = 0

        for record in iter_records(path):
            if not matches_score_filter(record, score_filter) or not matches_query(record, search):
                continue

            total += 1
            score = score_value(record)
            if score is not None:
                score_sum += score
                score_count += 1
                if score > 0:
                    correct += 1

            if total <= offset:
                continue
            if len(records) < limit:
                records.append(record)

        stats = {
            "correct": correct,
            "avg_score": (score_sum / score_count) if score_count else None,
        }
        json_response(self, {"records": records, "total": total, "offset": offset, "limit": limit, "stats": stats})


class RolloutViewerServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, data_dir: Path):
        super().__init__(server_address, request_handler_class)
        self.data_dir = data_dir.resolve()


def main() -> int:
    args = parse_args()
    data_dir_arg = args.data_dir_option or args.data_dir
    data_dir = Path(data_dir_arg).expanduser() if data_dir_arg else find_default_data_dir()
    if data_dir is None:
        print(
            "No data directory was provided, and no crrl_verl_pr/*/validation_data JSONL directory was found.",
            file=sys.stderr,
        )
        return 2

    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Data directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    server = RolloutViewerServer((args.host, args.port), RolloutViewerHandler, data_dir)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {data_dir}")
    print(f"Open {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
