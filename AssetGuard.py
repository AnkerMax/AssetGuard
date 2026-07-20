#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import requests

WEIGHTS = {
    "topic_match": 0.30,
    "detail_match": 0.20,
    "section_relevance": 0.20,
    "visual_evidence": 0.15,
    "contradictions": 0.15,
}

REQUEST_CONNECT_TIMEOUT = 10
REQUEST_READ_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 2
DEFAULT_REQUEST_DELAY = 1
DEFAULT_MAX_OUTPUT_TOKENS = 8000

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MEDIA_TYPES_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_STRICT_FAILURE = 3
EXIT_RUNTIME_ERROR = 4

LOGGER = logging.getLogger(__name__)

BACKEND_REQUIRED_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Backend compatibility placeholder tool. Must remain present even when unused.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rst_image_context_check",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "document_path": {"type": "string"},
                            "image_path": {"type": "string"},
                            "image_kind": {"type": "string", "enum": ["screenshot", "icon", "other"]},
                            "contains_interactive_buttons": {"type": "boolean"},
                            "buttons_magenta": {"type": "boolean"},
                            "hard_fail": {"type": "boolean"},
                            "hard_fail_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "criteria": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "topic_match": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "detail_match": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "section_relevance": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "visual_evidence": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "contradictions": {"type": "integer", "minimum": 0, "maximum": 3},
                                },
                                "required": ["topic_match", "detail_match", "section_relevance", "visual_evidence", "contradictions"],
                            },
                            "reasons": {"type": "array", "items": {"type": "string"}},
                            "missing_evidence": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["document_path", "image_path", "image_kind", "contains_interactive_buttons", "buttons_magenta", "hard_fail", "hard_fail_reason", "criteria", "reasons", "missing_evidence"],
                    },
                }
            },
            "required": ["results"],
        },
    },
}

@dataclass
class ImageReference:
    kind: str
    target: str
    line: int
    name: Optional[str] = None
    original_target: Optional[str] = None
    original_resolved_path: Optional[str] = None
    resolved_path: Optional[str] = None
    exists: bool = False
    is_valid_image: bool = False
    error: Optional[str] = None

@dataclass
class LoadedImage:
    path: str
    media_type: str
    data_base64: str

@dataclass
class ApiResult:
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    attached_image_count: int
    attached_images: List[str]
    http_status: Optional[int]
    finish_reason: Optional[str]
    attempt: int
    max_retries: int
    error: Optional[str] = None
    warning: Optional[str] = None
    response_excerpt: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class AuditRow:
    file_path: str
    title: Optional[str]
    image_count: int
    image_refs: List[Dict[str, Any]]
    result: Dict[str, Any]


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise SystemExit(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def compute_overall_score(criteria: Dict[str, int]) -> float:
    weighted = sum(criteria.get(k, 0) * WEIGHTS[k] for k in WEIGHTS)
    normalized = weighted / sum(3 * WEIGHTS[k] for k in WEIGHTS)
    return round(normalized, 2)


def verdict_from_score(score: float) -> str:
    if score >= 0.80:
        return "pass"
    if score >= 0.55:
        return "partial"
    return "fail"


def final_verdict(item: Dict[str, Any]) -> str:
    if item.get("hard_fail") is True:
        return "fail"
    return verdict_from_score(compute_overall_score(item.get("criteria", {})))


def extract_title(rst_raw: str) -> Optional[str]:
    lines = rst_raw.splitlines()
    adorn = set("=~-^\"'`:+*#<>")
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if title and underline and len(underline) >= len(title) and set(underline).issubset(adorn):
            return title
    return None


def extract_image_refs(rst_raw: str) -> List[ImageReference]:
    refs: List[ImageReference] = []
    patterns = [
        (r"^\s*\.\.\s+image::\s+(.+?)\s*$", "image"),
        (r"^\s*\.\.\s+figure::\s+(.+?)\s*$", "figure"),
        (r"^\s*\.\.\s+\|([^|]+)\|\s+image::\s+(.+?)\s*$", "substitution_image"),
    ]
    for idx, line in enumerate(rst_raw.splitlines(), start=1):
        for pattern, kind in patterns:
            match = re.match(pattern, line)
            if not match:
                continue
            if kind == "substitution_image":
                refs.append(ImageReference(kind=kind, name=match.group(1).strip(), target=match.group(2).strip(), line=idx))
            else:
                refs.append(ImageReference(kind=kind, target=match.group(1).strip(), line=idx))
    return refs


def normalize_target(target: str) -> str:
    return target.strip().strip('"').strip("'")


def get_image_suffix(path: str) -> str:
    return PurePosixPath(path).suffix.lower()


def is_valid_image_path(path: str) -> bool:
    return get_image_suffix(path) in VALID_IMAGE_SUFFIXES


def get_media_type_for_path(path: Path) -> str:
    media_type = MEDIA_TYPES_BY_SUFFIX.get(path.suffix.lower())
    if not media_type:
        raise ValueError("kein valides Bild")
    return media_type


def resolve_local_path(rst_file: Path, target: str, workspace: Path, source_root: Optional[Path] = None) -> Path:
    target = normalize_target(target)
    if target.startswith(("http://", "https://", "data:")):
        raise ValueError(f"Non-local image target found in RST: {target}")
    if target.startswith("/"):
        base = source_root if source_root is not None else workspace
        return (base / target.lstrip("/")).resolve()
    return (rst_file.parent / target).resolve()


def load_local_image_content(path: Path) -> LoadedImage:
    suffix = path.suffix.lower()
    if suffix not in VALID_IMAGE_SUFFIXES:
        raise ValueError("kein valides Bild")
    if not path.exists() or not path.is_file():
        raise ValueError("kein valides Bild")
    try:
        raw = path.read_bytes()
    except OSError:
        raise ValueError("kein valides Bild")
    return LoadedImage(path=str(path.resolve()), media_type=get_media_type_for_path(path), data_base64=base64.b64encode(raw).decode("utf-8"))


def build_image_candidates(rst_path: Path, refs: List[ImageReference], workspace: Path, source_root: Optional[Path] = None) -> List[ImageReference]:
    candidates: List[ImageReference] = []
    for ref in refs:
        try:
            resolved = resolve_local_path(rst_path, ref.target, workspace, source_root)
            valid = is_valid_image_path(ref.target)
            exists = resolved.exists()
            error = None if (valid and exists) else "kein valides Bild"
            resolved_path = str(resolved)
            original_resolved_path = str(resolved)
        except ValueError:
            valid = False
            exists = False
            error = "kein valides Bild"
            resolved_path = None
            original_resolved_path = None
        candidates.append(ImageReference(kind=ref.kind, name=ref.name, target=ref.target, line=ref.line, original_target=ref.target, original_resolved_path=original_resolved_path, resolved_path=resolved_path, exists=exists, is_valid_image=valid, error=error))
    return candidates


def make_prompt(job: Dict[str, Any]) -> str:
    image_count = job.get("attached_image_count", len(job.get("image_refs", [])))
    return (
        "Analyze the reStructuredText document and all attached images.\n"
        "Evaluate every attached image separately.\n"
        "Return one object in results for each attached image.\n\n"
        "First classify the image:\n"
        "- image_kind must be one of: screenshot, icon, other.\n"
        "- screenshot = UI/application/page screenshot with visible interface.\n"
        "- icon = small symbolic graphic, logo, pictogram, or isolated UI symbol.\n"
        "- other = anything else.\n\n"
        "Then apply this mandatory rule before contextual scoring:\n"
        "- If the image is a screenshot, check whether visible user-interactive buttons are magenta.\n"
        "- User-interactive buttons include clearly clickable UI controls such as buttons, CTA elements, or obvious interactive controls.\n"
        "- If the image is a screenshot and contains interactive buttons and those buttons are not magenta, set hard_fail=true.\n"
        "- In that case set hard_fail_reason to a short explanation.\n"
        "- In that case the image must be treated as failed regardless of the criteria scores.\n"
        "- If there are no visible interactive buttons, set contains_interactive_buttons=false and do not hard-fail for color.\n\n"
        "Scoring rules:\n"
        "- criteria scores still need to be filled for every image.\n"
        "- But if hard_fail=true, the final judgment must ignore the score.\n\n"
        "- criteria.topic_match: score from 0 to 3 for whether the main topic in the image matches the relevant rst content.\n"
        "- criteria.detail_match: score from 0 to 3 for whether important visual details match the rst content.\n"
        "- criteria.section_relevance: score from 0 to 3 for whether the image matches the most relevant section or context in the rst file.\n"
        "- criteria.visual_evidence: score from 0 to 3 for how clearly the image provides enough visible evidence for a reliable judgment.\n"
        "- criteria.contradictions: score from 0 to 3, where 3 means no clear contradiction (none) and 0 means strong contradiction with the rst content.\n"
        "- reasons: short bullet-style statements explaining the judgment.\n"
        "- missing_evidence: short bullet-style statements listing relevant information that is missing, unclear, or not visible enough for a stronger judgment.\n\n"
        "Scoring guidance:\n"
        "- 3 means strong / clear / fully supported.\n"
        "- 2 means mostly supported.\n"
        "- 1 means weakly supported or doubtful.\n"
        "- 0 means absent, not supported, or clearly contradictory.\n\n"
        "Field meaning:\n"
        "- document_path: use exactly the provided file path of the rst document.\n"
        "- image_path: use exactly the path that was provided for each attached image.\n"
        "- buttons_magenta should be true only if the visible interactive buttons are magenta.\n"
        "- If no interactive buttons are visible, set buttons_magenta=false.\n"
        "- reasons: short bullet-style statements explaining the judgment.\n"
        "- missing_evidence: short bullet-style statements listing missing or unclear information.\n\n"
        "Rules:\n"
        "- Use only the rst content and the attached image.\n"
        "- Do not guess facts that are not visible in the image or not stated in the rst.\n"
        "- Base the judgment on semantic relevance, not only keyword overlap.\n\n"
        f"FILE: {job['file_path']}\n"
        f"TITLE: {job.get('title') or ''}\n"
        f"ATTACHED_IMAGE_RELATIONS_IN_RST: {image_count}\n\n"
        "RST:\n<<<RST\n"
        f"{job['rst_raw']}\n"
        "RST>>>"
    )


def extract_response_text(data: Dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts: List[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(p for p in parts if p).strip()


def extract_finish_reason(data: Dict[str, Any]) -> Optional[str]:
    if isinstance(data.get("status"), str):
        return data["status"]
    for item in data.get("output", []):
        if isinstance(item, dict) and item.get("finish_reason"):
            return item["finish_reason"]
    return None


def collect_json_blocks(text: str, open_char: str, close_char: str) -> List[str]:
    blocks: List[str] = []
    start = text.find(open_char)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:i + 1])
                    break
        start = text.find(open_char, start + 1)
    return blocks


def _is_complete_result_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required_top = {"document_path", "image_path", "image_kind", "contains_interactive_buttons", "buttons_magenta", "hard_fail", "hard_fail_reason", "criteria", "reasons", "missing_evidence"}
    if not required_top.issubset(item.keys()):
        return False
    if item["image_kind"] not in {"screenshot", "icon", "other"}:
        return False
    if not isinstance(item.get("contains_interactive_buttons"), bool):
        return False
    if not isinstance(item.get("buttons_magenta"), bool):
        return False
    if not isinstance(item.get("hard_fail"), bool):
        return False
    if item.get("hard_fail_reason") is not None and not isinstance(item.get("hard_fail_reason"), str):
        return False
    criteria = item.get("criteria")
    if not isinstance(criteria, dict):
        return False
    required_criteria = {"topic_match", "detail_match", "section_relevance", "visual_evidence", "contradictions"}
    if not required_criteria.issubset(criteria.keys()):
        return False
    for key in required_criteria:
        value = criteria.get(key)
        if not isinstance(value, int) or value < 0 or value > 3:
            return False
    if not isinstance(item.get("reasons"), list) or not all(isinstance(x, str) for x in item.get("reasons")):
        return False
    if not isinstance(item.get("missing_evidence"), list) or not all(isinstance(x, str) for x in item.get("missing_evidence")):
        return False
    return True


def extract_response_json(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    for key in ("output_parsed", "parsed", "response_parsed"):
        candidate = data.get(key)
        if isinstance(candidate, dict):
            results = candidate.get("results")
            if isinstance(results, list) and all(_is_complete_result_item(x) for x in results):
                return candidate
    text = extract_response_text(data)
    if not text:
        return None
    candidates: List[Dict[str, Any]] = []
    for raw in collect_json_blocks(text, "{", "}"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            pass
    for raw in collect_json_blocks(text, "[", "]"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                candidates.append({"results": obj})
        except Exception:
            pass
    for obj in reversed(candidates):
        results = obj.get("results")
        if isinstance(results, list) and all(_is_complete_result_item(x) for x in results):
            return obj
    return None


class ResponsesClient:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def build_payload(self, prompt: str, images: List[LoadedImage], max_output_tokens: int) -> Tuple[Dict[str, Any], List[str]]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        attached_images: List[str] = []
        seen_paths = set()
        for idx, img in enumerate(images, start=1):
            if img.path in seen_paths:
                continue
            seen_paths.add(img.path)
            content.append({"type": "input_text", "text": f"IMAGE {idx} PATH: {img.path}"})
            content.append({"type": "input_image", "image_url": f"data:{img.media_type};base64,{img.data_base64}"})
            attached_images.append(img.path)
        payload = {
            "model": self.model,
            "instructions": (
                "You analyze reStructuredText documents and related images. "
                "Each image is preceded by a text line in the form 'IMAGE N PATH: <path>'. "
                "Use that exact path for the corresponding image. "
                "Do not guess paths. Return JSON only."
            ),
            "input": [{"role": "user", "content": content}],
            "tools": [BACKEND_REQUIRED_TOOL],
            "tool_choice": "none",
            "response_format": RESPONSE_SCHEMA,
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
        }
        return payload, attached_images

    def post_with_retries(self, payload: Dict[str, Any], attached_images: List[str], timeout: int = 180, max_retries: int = DEFAULT_MAX_RETRIES, request_delay: float = DEFAULT_REQUEST_DELAY) -> ApiResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_result: Optional[ApiResult] = None
        for attempt in range(1, max_retries + 1):
            if request_delay > 0:
                time.sleep(request_delay)
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=(REQUEST_CONNECT_TIMEOUT, min(timeout, REQUEST_READ_TIMEOUT)))
                status_code = response.status_code
                response_text = response.text
                if status_code in {429, 500, 502, 503, 504}:
                    last_result = ApiResult(raw_text="", parsed_json=None, attached_image_count=len(attached_images), attached_images=attached_images, http_status=status_code, finish_reason=None, attempt=attempt, max_retries=max_retries, error="backend_error", response_excerpt=response_text[:2000], raw_response=None)
                    LOGGER.warning("Transient backend error %s on attempt %s/%s", status_code, attempt, max_retries)
                    if attempt < max_retries:
                        continue
                    return last_result
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception:
                    data = {"_non_json_response_text": response_text}
                raw_text = extract_response_text(data) if isinstance(data, dict) else ""
                parsed_json = extract_response_json(data) if isinstance(data, dict) else None
                if raw_text:
                    LOGGER.info("Raw model response text (first 4000 chars):\n%s", raw_text[:4000])
                else:
                    LOGGER.warning("Model response contained no raw text.")
                if status_code == 200 and parsed_json is None:
                    LOGGER.warning("HTTP 200 received, but no valid structured JSON could be parsed from the model response.")
                return ApiResult(raw_text=raw_text, parsed_json=parsed_json, attached_image_count=len(attached_images), attached_images=attached_images, http_status=status_code, finish_reason=extract_finish_reason(data) if isinstance(data, dict) else None, attempt=attempt, max_retries=max_retries, response_excerpt=None if parsed_json else response_text[:2000], raw_response=data if isinstance(data, dict) else None)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_result = ApiResult(raw_text="", parsed_json=None, attached_image_count=len(attached_images), attached_images=attached_images, http_status=None, finish_reason=None, attempt=attempt, max_retries=max_retries, error="backend_error", response_excerpt=str(exc)[:2000], raw_response=None)
                LOGGER.warning("Connection/backend error on attempt %s/%s: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    continue
                return last_result
            except requests.RequestException as exc:
                response = getattr(exc, "response", None)
                return ApiResult(raw_text="", parsed_json=None, attached_image_count=len(attached_images), attached_images=attached_images, http_status=getattr(response, "status_code", None), finish_reason=None, attempt=attempt, max_retries=max_retries, error="backend_error", response_excerpt=(getattr(response, "text", str(exc)) or "")[:2000], raw_response=None)
        return last_result or ApiResult(raw_text="", parsed_json=None, attached_image_count=len(attached_images), attached_images=attached_images, http_status=None, finish_reason=None, attempt=max_retries, max_retries=max_retries, error="backend_error", raw_response=None)

    def analyze_images(self, prompt: str, images: List[LoadedImage], timeout: int = 180, max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, request_delay: float = DEFAULT_REQUEST_DELAY) -> ApiResult:
        payload, attached_images = self.build_payload(prompt, images, max_output_tokens)
        return self.post_with_retries(payload=payload, attached_images=attached_images, timeout=timeout, max_retries=max_retries, request_delay=request_delay)


def read_file_list(file_list: Path) -> List[Path]:
    items = []
    for line in file_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(Path(line))
    return items


def find_rst_files(workspace: Path, path_prefixes: List[str]) -> List[Path]:
    files = []
    for path in workspace.rglob("*.rst"):
        rel = path.relative_to(workspace).as_posix()
        if path_prefixes and not any(rel.startswith(prefix) for prefix in path_prefixes):
            continue
        files.append(path)
    return sorted(files)


def select_input_files(args: argparse.Namespace, workspace: Path) -> List[Path]:
    if args.rst_file:
        files = []
        for p in args.rst_file:
            path = Path(p).expanduser()
            files.append((workspace / path).resolve() if not path.is_absolute() else path.resolve())
    elif args.file_list:
        files = []
        for p in read_file_list(Path(args.file_list).expanduser()):
            path = Path(p).expanduser()
            files.append((workspace / path).resolve() if not path.is_absolute() else path.resolve())
    else:
        files = find_rst_files(workspace, args.path_prefix)
    deduped: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved.suffix.lower() != ".rst":
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)
    return deduped


def make_row(rst_file: Path, workspace: Path, title: Optional[str], image_refs: List[ImageReference], result: ApiResult) -> AuditRow:
    rel_path = rst_file.relative_to(workspace).as_posix() if rst_file.is_relative_to(workspace) else str(rst_file)
    return AuditRow(file_path=rel_path, title=title, image_count=len(image_refs), image_refs=[{"original_target": img.original_target, "original_path": img.original_resolved_path, "used_path": img.resolved_path, "kind": img.kind, "line": img.line, "exists": img.exists, "is_valid_image": img.is_valid_image, "error": img.error} for img in image_refs], result=asdict(result))


def build_result_items_for_json(row: AuditRow) -> List[Dict[str, Any]]:
    result = row.result or {}
    parsed = result.get("parsed_json") or {}
    items = parsed.get("results") if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        items = []
    enriched: List[Dict[str, Any]] = []
    for item in items:
        criteria = item.get("criteria", {})
        enriched.append({**item, "overall_score": compute_overall_score(criteria), "verdict": final_verdict(item)})
    return enriched


def build_json_row(row: AuditRow) -> Dict[str, Any]:
    result = row.result or {}
    enriched_results = build_result_items_for_json(row)
    verdicts = [item.get("verdict") for item in enriched_results]
    summary = {"pass": sum(1 for v in verdicts if v == "pass"), "partial": sum(1 for v in verdicts if v == "partial"), "fail": sum(1 for v in verdicts if v == "fail")}
    return {"file_path": row.file_path, "title": row.title, "image_count": row.image_count, "image_refs": row.image_refs, "status": {"http_status": result.get("http_status"), "finish_reason": result.get("finish_reason"), "attempt": result.get("attempt"), "max_retries": result.get("max_retries"), "error": result.get("error"), "warning": result.get("warning"), "attached_image_count": result.get("attached_image_count")}, "summary": summary, "results": enriched_results, "response_excerpt": result.get("response_excerpt"), "raw_response": result.get("raw_response")}


def iter_csv_rows(row: AuditRow) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    invalid_refs = [ref for ref in row.image_refs if ref.get("error") == "kein valides Bild"]
    if invalid_refs:
        for ref in invalid_refs:
            output_rows.append({"document_file": row.file_path, "document_title": row.title or "", "image_file": ref.get("used_path") or ref.get("original_path") or "", "image_reference_type": ref.get("kind") or "", "image_reference_line": ref.get("line") or "", "detected_image_type": "", "has_interactive_buttons": "", "interactive_buttons_magenta": "", "hard_fail_triggered": "", "hard_fail_reason": "", "score_topic_match": "", "score_detail_match": "", "score_section_relevance": "", "score_visual_evidence": "", "score_contradictions": "", "overall_score": "", "final_verdict": "fail", "processing_error": "kein valides Bild", "api_http_status": row.result.get("http_status"), "api_finish_reason": row.result.get("finish_reason"), "api_attempt": row.result.get("attempt"), "match_reasons": "", "missing_evidence": ""})
    if row.result.get("error") == "backend_error":
        output_rows.append({"document_file": row.file_path, "document_title": row.title or "", "image_file": "", "image_reference_type": "", "image_reference_line": "", "detected_image_type": "", "has_interactive_buttons": "", "interactive_buttons_magenta": "", "hard_fail_triggered": "", "hard_fail_reason": "", "score_topic_match": "", "score_detail_match": "", "score_section_relevance": "", "score_visual_evidence": "", "score_contradictions": "", "overall_score": "", "final_verdict": "fail", "processing_error": "backend_error", "api_http_status": row.result.get("http_status"), "api_finish_reason": row.result.get("finish_reason"), "api_attempt": row.result.get("attempt"), "match_reasons": "", "missing_evidence": ""})
        return output_rows
    parsed = row.result.get("parsed_json") or {}
    results = parsed.get("results", [])
    if not isinstance(results, list):
        return output_rows
    ref_by_used_path = {ref.get("used_path"): ref for ref in row.image_refs if ref.get("used_path")}
    for item in results:
        criteria = item.get("criteria", {})
        image_path = item.get("image_path", "")
        ref = ref_by_used_path.get(image_path, {})
        output_rows.append({"document_file": row.file_path, "document_title": row.title or "", "image_file": image_path, "image_reference_type": ref.get("kind", ""), "image_reference_line": ref.get("line", ""), "detected_image_type": item.get("image_kind"), "has_interactive_buttons": item.get("contains_interactive_buttons"), "interactive_buttons_magenta": item.get("buttons_magenta"), "hard_fail_triggered": item.get("hard_fail"), "hard_fail_reason": item.get("hard_fail_reason") or "", "score_topic_match": criteria.get("topic_match"), "score_detail_match": criteria.get("detail_match"), "score_section_relevance": criteria.get("section_relevance"), "score_visual_evidence": criteria.get("visual_evidence"), "score_contradictions": criteria.get("contradictions"), "overall_score": compute_overall_score(criteria), "final_verdict": final_verdict(item), "processing_error": row.result.get("error") or "", "api_http_status": row.result.get("http_status"), "api_finish_reason": row.result.get("finish_reason"), "api_attempt": row.result.get("attempt"), "match_reasons": " | ".join(item.get("reasons", [])), "missing_evidence": " | ".join(item.get("missing_evidence", []))})
    return output_rows


def write_csv(csv_output: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["document_file", "document_title", "image_file", "image_reference_type", "image_reference_line", "detected_image_type", "has_interactive_buttons", "interactive_buttons_magenta", "hard_fail_triggered", "hard_fail_reason", "score_topic_match", "score_detail_match", "score_section_relevance", "score_visual_evidence", "score_contradictions", "overall_score", "final_verdict", "processing_error", "api_http_status", "api_finish_reason", "api_attempt", "match_reasons", "missing_evidence"]
    with csv_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def filter_failed_csv_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("final_verdict") == "fail"]


def process_file(rst_file: Path, workspace: Path, source_root: Optional[Path], client: ResponsesClient, max_retries: int, request_delay: float, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> Optional[AuditRow]:
    if not rst_file.exists() or not rst_file.is_file():
        LOGGER.warning("Skipping missing file: %s", rst_file)
        return None
    rst_raw = rst_file.read_text(encoding="utf-8", errors="replace")
    refs = extract_image_refs(rst_raw)
    image_refs = build_image_candidates(rst_file, refs, workspace, source_root)
    if not image_refs:
        LOGGER.debug("No image references found in %s", rst_file)
        return None
    loaded_images: List[LoadedImage] = []
    seen_paths = set()
    for ref in image_refs:
        if ref.error == "kein valides Bild" or not ref.resolved_path:
            continue
        try:
            loaded = load_local_image_content(Path(ref.resolved_path))
        except ValueError:
            ref.error = "kein valides Bild"
            continue
        if loaded.path in seen_paths:
            continue
        seen_paths.add(loaded.path)
        loaded_images.append(loaded)
    rel_path = rst_file.relative_to(workspace).as_posix() if rst_file.is_relative_to(workspace) else str(rst_file)
    job = {"file_path": rel_path, "title": extract_title(rst_raw), "rst_raw": rst_raw, "image_refs": image_refs}
    prompt = make_prompt(job)
    if not loaded_images:
        LOGGER.warning("No valid images for %s", rst_file)
        result = ApiResult(raw_text="", parsed_json=None, attached_image_count=0, attached_images=[], http_status=None, finish_reason=None, attempt=0, max_retries=max_retries, error="kein valides Bild", response_excerpt="kein valides Bild", raw_response=None)
        return make_row(rst_file, workspace, job["title"], image_refs, result)
    LOGGER.info("Analyzing %s with %d attached images", rel_path, len(loaded_images))
    try:
        result = client.analyze_images(prompt=prompt, images=loaded_images, max_retries=max_retries, request_delay=request_delay, max_output_tokens=max_output_tokens)
    except Exception as exc:
        LOGGER.exception("Unexpected analyze_images error for %s", rel_path)
        result = ApiResult(raw_text="", parsed_json=None, attached_image_count=len(loaded_images), attached_images=[img.path for img in loaded_images], http_status=None, finish_reason=None, attempt=max_retries, max_retries=max_retries, error="backend_error", response_excerpt=str(exc)[:2000], raw_response=None)
    return make_row(rst_file, workspace, job["title"], image_refs, result)


def process_files(files: List[Path], workspace: Path, source_root: Optional[Path], client: ResponsesClient, json_output: Path, csv_output: Path, failed_csv_output: Path, max_retries: int, request_delay: float, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> Tuple[int, int, int]:
    processed_files = 0
    flagged_files = 0
    all_json_rows: List[Dict[str, Any]] = []
    all_csv_rows: List[Dict[str, Any]] = []
    for rst_file in files:
        row = process_file(rst_file=rst_file, workspace=workspace, source_root=source_root, client=client, max_retries=max_retries, request_delay=request_delay, max_output_tokens=max_output_tokens)
        if row is None:
            continue
        processed_files += 1
        json_row = build_json_row(row)
        all_json_rows.append(json_row)
        csv_rows = iter_csv_rows(row)
        all_csv_rows.extend(csv_rows)
        has_flagged_result = any(r.get("verdict") in {"partial", "fail"} for r in json_row.get("results", []))
        has_errors = any(ref.get("error") == "kein valides Bild" for ref in row.image_refs) or row.result.get("error") in {"kein valides Bild", "backend_error"}
        if has_flagged_result or has_errors:
            flagged_files += 1
    json_output.write_text(json.dumps(all_json_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_output, all_csv_rows)
    failed_csv_rows = filter_failed_csv_rows(all_csv_rows)
    write_csv(failed_csv_output, failed_csv_rows)
    return processed_files, flagged_files, len(all_json_rows)


def enforce_strict_mode(json_output: Path, fail_on_partial: bool = False) -> None:
    if not json_output.exists():
        raise SystemExit(EXIT_STRICT_FAILURE)
    data = json.loads(json_output.read_text(encoding="utf-8"))
    for row in data:
        image_refs = row.get("image_refs", [])
        if any(ref.get("error") == "kein valides Bild" for ref in image_refs):
            raise SystemExit(EXIT_STRICT_FAILURE)
        status = row.get("status") or {}
        if status.get("error") in {"kein valides Bild", "backend_error"}:
            raise SystemExit(EXIT_STRICT_FAILURE)
        results = row.get("results")
        if not isinstance(results, list):
            raise SystemExit(EXIT_STRICT_FAILURE)
        for item in results:
            if not _is_complete_result_item(item):
                raise SystemExit(EXIT_STRICT_FAILURE)
            verdict = item.get("verdict")
            if item.get("hard_fail") is True:
                raise SystemExit(EXIT_STRICT_FAILURE)
            if verdict == "fail":
                raise SystemExit(EXIT_STRICT_FAILURE)
            if fail_on_partial and verdict == "partial":
                raise SystemExit(EXIT_STRICT_FAILURE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit .rst image references with structured model output.")
    parser.add_argument("--workspace", default=".", help="Local repo/workspace path. Defaults to current directory.")
    parser.add_argument("--source-root", default=None, help="Documentation source root for leading-slash image paths.")
    parser.add_argument("--file-list", default=None, help="Text file with one repo-relative .rst path per line.")
    parser.add_argument("--rst-file", action="append", default=[], help="Single .rst file to process. Can be used multiple times.")
    parser.add_argument("--path-prefix", action="append", default=[], help="Only process .rst files whose path starts with this prefix.")
    parser.add_argument("--api-url", default=os.getenv("AI_API_URL"), help="Responses endpoint, e.g. .../v1/responses")
    parser.add_argument("--api-key", default=os.getenv("AI_API_KEY"), help="AI API key.")
    parser.add_argument("--model", default=os.getenv("AI_MODEL", "qwen3.6-35b"), help="Model name.")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY, help="Fixed delay before each API call.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Number of attempts for backend/transient errors.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Maximum output tokens for the API.")
    parser.add_argument("--output-json", default="results_with_images.json", help="Machine-readable JSON output.")
    parser.add_argument("--output-csv", default="results_with_images.csv", help="Flat CSV output.")
    parser.add_argument("--output-failed-csv", default="results_with_images.failed_only.csv", help="CSV output containing only failed results.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on strict validation failure.")
    parser.add_argument("--fail-on-partial", action="store_true", help="In strict mode, also fail when verdict is partial.")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        if not args.api_url:
            LOGGER.error("Missing AI API URL. Use --api-url or set AI_API_URL.")
            raise SystemExit(EXIT_CONFIG_ERROR)
        if not args.api_key:
            LOGGER.error("Missing AI API key. Use --api-key or set AI_API_KEY.")
            raise SystemExit(EXIT_CONFIG_ERROR)
        workspace = Path(args.workspace).expanduser().resolve()
        source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
        files = select_input_files(args, workspace)
        LOGGER.info("Selected %d rst files", len(files))
        client = ResponsesClient(api_url=args.api_url, api_key=args.api_key, model=args.model)
        processed_files, flagged_files, row_count = process_files(files=files, workspace=workspace, source_root=source_root, client=client, json_output=Path(args.output_json), csv_output=Path(args.output_csv), failed_csv_output=Path(args.output_failed_csv), max_retries=args.max_retries, request_delay=args.request_delay, max_output_tokens=args.max_output_tokens)
        if args.strict:
            enforce_strict_mode(Path(args.output_json), fail_on_partial=args.fail_on_partial)
        LOGGER.info("Done. Processed %d rst files, wrote %d rows to %s, flagged %d files, csv=%s, failed_csv=%s", processed_files, row_count, args.output_json, flagged_files, args.output_csv, args.output_failed_csv)
        raise SystemExit(EXIT_OK)
    except SystemExit:
        raise
    except Exception:
        LOGGER.exception("Unhandled runtime error")
        raise SystemExit(EXIT_RUNTIME_ERROR)


if __name__ == "__main__":
    main()
