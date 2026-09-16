#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import logging
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from json import JSONDecoder, JSONDecodeError
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

WEIGHTS = {
    "topic_match": 0.30,
    "detail_match": 0.20,
    "section_relevance": 0.20,
    "visual_evidence": 0.15,
    "contradictions": 0.15,
}

REQUEST_CONNECT_TIMEOUT = 10
REQUEST_READ_TIMEOUT = 240
DEFAULT_MAX_RETRIES = 2
DEFAULT_REQUEST_DELAY = 1
DEFAULT_MAX_OUTPUT_TOKENS = 8000

DEFAULT_ORG = "opentelekomcloud-docs"
DEFAULT_REPO_LIMIT = 105
DEFAULT_MAX_WORKERS = 8
DEFAULT_WORKER_START_DELAY = 0.5

# Change this to the exact color that must trigger a local hard fail.
FORBIDDEN_COLOR_HEX = "#3298ff"
# 0 = exact RGB match. Use a small value such as 5-10 for JPEG artifacts if needed.
FORBIDDEN_COLOR_TOLERANCE = 0

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MEDIA_TYPES_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

BACKEND_REQUIRED_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Backend compatibility placeholder tool. Must remain present even when unused.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
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
                            "image_kind": {
                                "type": "string",
                                "enum": ["screenshot", "icon", "other"],
                            },
                            "contains_interactive_buttons": {"type": "boolean"},
                            "buttons_magenta": {"type": "boolean"},
                            "hard_fail": {"type": "boolean"},
                            "hard_fail_reason": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
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
                                "required": [
                                    "topic_match",
                                    "detail_match",
                                    "section_relevance",
                                    "visual_evidence",
                                    "contradictions",
                                ],
                            },
                            "reasons": {"type": "array", "items": {"type": "string"}},
                            "missing_evidence": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "document_path",
                            "image_path",
                            "image_kind",
                            "contains_interactive_buttons",
                            "buttons_magenta",
                            "hard_fail",
                            "hard_fail_reason",
                            "criteria",
                            "reasons",
                            "missing_evidence",
                        ],
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
    raw_response: Optional[Dict[str, Any]]
    http_status: Optional[int]
    http_response_text: str
    finish_reason: Optional[str]
    attempt: int
    max_retries: int
    error: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class AuditRow:
    file_path: str
    title: Optional[str]
    image_count: int
    image_refs: List[Dict[str, Any]]
    result: Dict[str, Any]


def compute_overall_score(criteria: Dict[str, int]) -> float:
    weighted = sum(criteria.get(key, 0) * WEIGHTS[key] for key in WEIGHTS)
    normalized = weighted / sum(3 * WEIGHTS[key] for key in WEIGHTS)
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


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def image_contains_color(path: Path, target_rgb: Tuple[int, int, int], tolerance: int = 0) -> bool:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        for pixel in rgb_image.getdata():
            if all(abs(pixel[index] - target_rgb[index]) <= tolerance for index in range(3)):
                return True
    return False


def extract_title(rst_raw: str) -> Optional[str]:
    lines = rst_raw.splitlines()
    adorn = set("=~-^\"'`:+*#<>")
    for index in range(len(lines) - 1):
        title = lines[index].strip()
        underline = lines[index + 1].strip()
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
    for line_number, line in enumerate(rst_raw.splitlines(), start=1):
        for pattern, kind in patterns:
            match = re.match(pattern, line)
            if not match:
                continue
            if kind == "substitution_image":
                refs.append(ImageReference(kind=kind, name=match.group(1).strip(), target=match.group(2).strip(), line=line_number))
            else:
                refs.append(ImageReference(kind=kind, target=match.group(1).strip(), line=line_number))
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
        raise ValueError("invalid image")
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
    if path.suffix.lower() not in VALID_IMAGE_SUFFIXES or not path.exists() or not path.is_file():
        raise ValueError("invalid image")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("invalid image") from exc
    return LoadedImage(
        path=str(path.resolve()),
        media_type=get_media_type_for_path(path),
        data_base64=base64.b64encode(raw).decode("utf-8"),
    )


def build_image_candidates(rst_path: Path, refs: List[ImageReference], workspace: Path, source_root: Optional[Path] = None) -> List[ImageReference]:
    candidates: List[ImageReference] = []
    for ref in refs:
        try:
            resolved = resolve_local_path(rst_path, ref.target, workspace, source_root)
            valid = is_valid_image_path(ref.target)
            exists = resolved.exists()
            candidates.append(ImageReference(
                kind=ref.kind,
                name=ref.name,
                target=ref.target,
                line=ref.line,
                original_target=ref.target,
                original_resolved_path=str(resolved),
                resolved_path=str(resolved),
                exists=exists,
                is_valid_image=valid,
                error=None if valid and exists else "invalid image",
            ))
        except ValueError:
            candidates.append(ImageReference(
                kind=ref.kind,
                name=ref.name,
                target=ref.target,
                line=ref.line,
                original_target=ref.target,
                exists=False,
                is_valid_image=False,
                error="invalid image",
            ))
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
        "Do not evaluate colors and do not use color as a hard-fail criterion.\n"
        "Color validation is performed locally before this API request.\n"
        "Always set hard_fail=false and hard_fail_reason=null.\n"
        "Set contains_interactive_buttons based only on visibility of interactive controls.\n"
        "Set buttons_magenta=false.\n\n"
        "Scoring rules:\n"
        "- criteria scores need to be filled for every image.\n"
        "- criteria.topic_match: score from 0 to 3.\n"
        "- criteria.detail_match: score from 0 to 3.\n"
        "- criteria.section_relevance: score from 0 to 3.\n"
        "- criteria.visual_evidence: score from 0 to 3.\n"
        "- criteria.contradictions: score from 0 to 3, where 3 means no clear contradiction.\n"
        "- reasons: short bullet-style statements explaining the judgment.\n"
        "- missing_evidence: short bullet-style statements listing missing or unclear information.\n\n"
        "Output rules:\n"
        "- Return JSON only.\n"
        "- Use exactly the schema fields.\n"
        "- document_path must use exactly the provided rst file path.\n"
        "- image_path must use exactly the provided image path.\n"
        "- Do not add markdown fences.\n"
        "- Do not add analysis text before or after the JSON.\n\n"
        "Evidence rules:\n"
        "- Use only the rst content and the attached image.\n"
        "- Do not guess facts that are not visible in the image or not stated in the rst.\n"
        "- Base the judgment on semantic relevance, not only keyword overlap.\n\n"
        f"FILE: {job['file_path']}\n"
        f"TITLE: {job.get('title') or ''}\n"
        f"ATTACHED_IMAGE_RELATIONS_IN_RST: {image_count}\n\n"
        f"RST:\n{job['rst_raw']}"
    )


def extract_finish_reason(data: Dict[str, Any]) -> Optional[str]:
    if isinstance(data.get("status"), str):
        return data["status"]
    for item in data.get("output", []):
        if isinstance(item, dict) and item.get("finish_reason"):
            return item["finish_reason"]
    return None


def _is_complete_result_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    required_top = {
        "document_path", "image_path", "image_kind", "contains_interactive_buttons",
        "buttons_magenta", "hard_fail", "hard_fail_reason", "criteria", "reasons", "missing_evidence",
    }
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
    if not isinstance(item.get("reasons"), list) or not all(isinstance(x, str) for x in item["reasons"]):
        return False
    if not isinstance(item.get("missing_evidence"), list) or not all(isinstance(x, str) for x in item["missing_evidence"]):
        return False
    return True


def _normalize_candidate(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"results": obj}
    return None


def extract_response_text(data: Dict[str, Any]) -> str:
    parts: List[str] = []
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        parts.append(output_text.strip())
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                    text = part["text"].strip()
                    if text:
                        parts.append(text)
    seen = set()
    deduped = []
    for part in parts:
        key = part[:500]
        if key not in seen:
            seen.add(key)
            deduped.append(part)
    return "\n".join(deduped).strip()


def extract_response_json(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    candidates: List[Dict[str, Any]] = []

    def add_candidate(obj: Any) -> None:
        normalized = _normalize_candidate(obj)
        if normalized is not None:
            candidates.append(normalized)

    def scan_text(text: str) -> None:
        decoder = JSONDecoder()
        index = 0
        while index < len(text):
            if text[index] not in "{[":
                index += 1
                continue
            try:
                obj, end = decoder.raw_decode(text, index)
                add_candidate(obj)
                index = max(index + 1, end)
            except JSONDecodeError:
                index += 1

    def walk_output(container: Any) -> None:
        if not isinstance(container, list):
            return
        for item in container:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_json", "json"}:
                    add_candidate(part.get("json"))
                elif part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                    scan_text(part["text"])

    for key in ("output_parsed", "parsed", "response_parsed"):
        add_candidate(data.get(key))
    if isinstance(data.get("output_text"), str):
        scan_text(data["output_text"])
    walk_output(data.get("output"))
    response_obj = data.get("response")
    if isinstance(response_obj, dict):
        if isinstance(response_obj.get("output_text"), str):
            scan_text(response_obj["output_text"])
        walk_output(response_obj.get("output"))
    for candidate in reversed(candidates):
        results = candidate.get("results")
        if isinstance(results, list) and results and all(_is_complete_result_item(item) for item in results):
            return candidate
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
        for index, image in enumerate(images, start=1):
            if image.path in seen_paths:
                continue
            seen_paths.add(image.path)
            content.append({"type": "input_text", "text": f"IMAGE {index} PATH: {image.path}"})
            content.append({"type": "input_image", "image_url": f"data:{image.media_type};base64,{image.data_base64}"})
            attached_images.append(image.path)
        payload = {
            "model": self.model,
            "instructions": (
                "You analyze reStructuredText documents and related images. "
                "Each image is preceded by a text line in the form 'IMAGE N PATH: '. "
                "Use that exact path for the corresponding image. "
                "Do not guess paths. Return JSON only."
            ),
            "input": [{"role": "user", "content": content}],
            "tools": [BACKEND_REQUIRED_TOOL],
            "tool_choice": "none",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": RESPONSE_SCHEMA["json_schema"]["name"],
                    "strict": RESPONSE_SCHEMA["json_schema"]["strict"],
                    "schema": RESPONSE_SCHEMA["json_schema"]["schema"],
                }
            },
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
                    last_result = ApiResult("", None, len(attached_images), attached_images, None, status_code, response_text, None, attempt, max_retries, error="backend_error")
                    if attempt < max_retries:
                        continue
                    return last_result
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception:
                    data = {"_non_json_response_text": response_text}
                return ApiResult(
                    raw_text=extract_response_text(data) if isinstance(data, dict) else "",
                    parsed_json=extract_response_json(data) if isinstance(data, dict) else None,
                    attached_image_count=len(attached_images),
                    attached_images=attached_images,
                    raw_response=data if isinstance(data, dict) else None,
                    http_status=status_code,
                    http_response_text=response_text,
                    finish_reason=extract_finish_reason(data) if isinstance(data, dict) else None,
                    attempt=attempt,
                    max_retries=max_retries,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_result = ApiResult("", None, len(attached_images), attached_images, None, None, str(exc), None, attempt, max_retries, error="backend_error")
                if attempt < max_retries:
                    continue
                return last_result
            except requests.RequestException as exc:
                return ApiResult("", None, len(attached_images), attached_images, None, getattr(getattr(exc, "response", None), "status_code", None), getattr(getattr(exc, "response", None), "text", str(exc)), None, attempt, max_retries, error="backend_error")
        return last_result or ApiResult("", None, len(attached_images), attached_images, None, None, "", None, max_retries, max_retries, error="backend_error")

    def analyze_images(self, prompt: str, images: List[LoadedImage], timeout: int = 180, max_retries: int = DEFAULT_MAX_RETRIES, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, request_delay: float = DEFAULT_REQUEST_DELAY) -> ApiResult:
        payload, attached_images = self.build_payload(prompt, images, max_output_tokens)
        return self.post_with_retries(payload, attached_images, timeout, max_retries, request_delay)


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
        relative_path = path.relative_to(workspace).as_posix()
        if path_prefixes and not any(relative_path.startswith(prefix) for prefix in path_prefixes):
            continue
        files.append(path)
    return sorted(files)


def select_input_files(args: argparse.Namespace, workspace: Path) -> List[Path]:
    if args.rst_file:
        files = []
        for raw_path in args.rst_file:
            path = Path(raw_path).expanduser()
            files.append((workspace / path).resolve() if not path.is_absolute() else path.resolve())
    elif args.file_list:
        files = []
        for raw_path in read_file_list(Path(args.file_list).expanduser()):
            files.append((workspace / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve())
    else:
        files = find_rst_files(workspace, args.path_prefix)
    deduped: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved.suffix.lower() != ".rst":
            continue
        if str(resolved) not in seen:
            seen.add(str(resolved))
            deduped.append(resolved)
    return deduped


def make_row(rst_file: Path, workspace: Path, title: Optional[str], image_refs: List[ImageReference], result: ApiResult) -> AuditRow:
    relative_path = rst_file.relative_to(workspace).as_posix() if rst_file.is_relative_to(workspace) else str(rst_file)
    return AuditRow(
        file_path=relative_path,
        title=title,
        image_count=len(image_refs),
        image_refs=[{
            "original_target": image.original_target,
            "original_path": image.original_resolved_path,
            "used_path": image.resolved_path,
            "kind": image.kind,
            "line": image.line,
            "exists": image.exists,
            "is_valid_image": image.is_valid_image,
            "error": image.error,
        } for image in image_refs],
        result=asdict(result),
    )


def local_hard_fail_item(document_path: str, image_path: str) -> Dict[str, Any]:
    return {
        "document_path": document_path,
        "image_path": image_path,
        "image_kind": "other",
        "contains_interactive_buttons": False,
        "buttons_magenta": False,
        "hard_fail": True,
        "hard_fail_reason": f"Forbidden color {FORBIDDEN_COLOR_HEX} was detected in the image.",
        "criteria": {
            "topic_match": 0,
            "detail_match": 0,
            "section_relevance": 0,
            "visual_evidence": 0,
            "contradictions": 0,
        },
        "reasons": [f"Local color check detected {FORBIDDEN_COLOR_HEX}."],
        "missing_evidence": [],
    }


def process_file(rst_file: Path, workspace: Path, source_root: Optional[Path], client: ResponsesClient, max_retries: int, request_delay: float, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> Optional[AuditRow]:
    if not rst_file.exists() or not rst_file.is_file():
        return None
    rst_raw = rst_file.read_text(encoding="utf-8", errors="replace")
    refs = extract_image_refs(rst_raw)
    image_refs = build_image_candidates(rst_file, refs, workspace, source_root)
    if not image_refs:
        return None

    relative_document_path = rst_file.relative_to(workspace).as_posix() if rst_file.is_relative_to(workspace) else str(rst_file)
    job = {
        "file_path": relative_document_path,
        "title": extract_title(rst_raw),
        "rst_raw": rst_raw,
        "image_refs": image_refs,
    }
    target_rgb = hex_to_rgb(FORBIDDEN_COLOR_HEX)
    local_failures: List[Dict[str, Any]] = []
    loaded_images: List[LoadedImage] = []
    seen_paths = set()

    for ref in image_refs:
        if ref.error == "invalid image" or not ref.resolved_path:
            continue
        image_path = Path(ref.resolved_path)
        try:
            if image_contains_color(image_path, target_rgb, FORBIDDEN_COLOR_TOLERANCE):
                local_failures.append(local_hard_fail_item(relative_document_path, str(image_path.resolve())))
                continue
            loaded = load_local_image_content(image_path)
        except (ValueError, OSError) as exc:
            ref.error = f"local image validation failed: {exc}"
            continue
        if loaded.path not in seen_paths:
            seen_paths.add(loaded.path)
            loaded_images.append(loaded)

    # Only images that do not contain the forbidden color are sent to the API.
    api_result: Optional[ApiResult] = None
    if loaded_images:
        try:
            api_result = client.analyze_images(
                prompt=make_prompt(job),
                images=loaded_images,
                max_retries=max_retries,
                request_delay=request_delay,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            api_result = ApiResult("", None, len(loaded_images), [image.path for image in loaded_images], None, None, str(exc), None, max_retries, max_retries, error="backend_error")

    if api_result is None:
        if local_failures:
            result = ApiResult(
                raw_text="",
                parsed_json={"results": local_failures},
                attached_image_count=0,
                attached_images=[],
                raw_response=None,
                http_status=None,
                http_response_text="local color hard fail",
                finish_reason="local_hard_fail",
                attempt=0,
                max_retries=max_retries,
                warning="Images with the forbidden color were not sent to the AI API.",
            )
        else:
            result = ApiResult("", None, 0, [], None, None, "invalid image", None, 0, max_retries, error="invalid image")
    else:
        parsed = api_result.parsed_json or {"results": []}
        api_results = parsed.get("results", []) if isinstance(parsed, dict) else []
        api_result.parsed_json = {"results": local_failures + api_results}
        if local_failures:
            api_result.warning = "Images with the forbidden color were not sent to the AI API."
        result = api_result

    return make_row(rst_file, workspace, job["title"], image_refs, result)


def result_label(verdict: str) -> str:
    labels = {"pass": "Pass", "partial": "Review recommended", "fail": "Fail"}
    return labels.get(verdict, "Unknown")


def finding_text(item: Dict[str, Any], verdict: str) -> str:
    if item.get("hard_fail") is True:
        return item.get("hard_fail_reason") or "The image violates a mandatory rule."
    reasons = item.get("reasons", [])
    if reasons:
        return " ".join(reasons)
    if verdict == "partial":
        return "The contextual or visual suitability of the image should be reviewed."
    if verdict == "fail":
        return "The image does not meet the audit criteria."
    return "No issue detected."


def recommendation_text(item: Dict[str, Any], verdict: str) -> str:
    if item.get("hard_fail") is True:
        return "Replace the image or remove the forbidden color."
    if verdict == "fail":
        return "Replace or rework the image to meet content and visual requirements."
    if verdict == "partial":
        return "Manually review the image and adapt it if necessary."
    return "No action required."


def image_path_for_output(image_path: str, workspace: Path) -> str:
    try:
        return Path(image_path).resolve().relative_to(workspace).as_posix()
    except (ValueError, OSError):
        return image_path


def build_csv_rows(row: AuditRow, workspace: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    parsed = ((row.result or {}).get("parsed_json")) or {}
    parsed_results = parsed.get("results", []) if isinstance(parsed, dict) else []
    for item in parsed_results:
        verdict = final_verdict(item)
        if verdict == "pass":
            continue
        rows.append({
            "Document": row.file_path,
            "Title": row.title or "",
            "Image": image_path_for_output(item.get("image_path", ""), workspace),
            "Result": result_label(verdict),
            "Finding": finding_text(item, verdict),
            "Recommended action": recommendation_text(item, verdict),
        })
    return rows


def build_json_row(row: AuditRow) -> Dict[str, Any]:
    parsed = ((row.result or {}).get("parsed_json")) or {}
    parsed_results = parsed.get("results", []) if isinstance(parsed, dict) else []
    enriched_results = []
    summary = {"pass": 0, "partial": 0, "fail": 0}
    for item in parsed_results:
        score = compute_overall_score(item.get("criteria", {}))
        verdict = final_verdict(item)
        summary[verdict] += 1
        enriched_item = dict(item)
        enriched_item["overall_score"] = score
        enriched_item["verdict"] = verdict
        enriched_results.append(enriched_item)
    return {
        "file_path": row.file_path,
        "title": row.title,
        "image_count": row.image_count,
        "image_refs": row.image_refs,
        "status": {
            "http_status": row.result.get("http_status"),
            "finish_reason": row.result.get("finish_reason"),
            "attempt": row.result.get("attempt"),
            "max_retries": row.result.get("max_retries"),
            "error": row.result.get("error"),
            "warning": row.result.get("warning"),
            "attached_image_count": row.result.get("attached_image_count"),
        },
        "summary": summary,
        "results": enriched_results,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["Document", "Title", "Image", "Result", "Finding", "Recommended action"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_files(files: List[Path], workspace: Path, source_root: Optional[Path], client: ResponsesClient, json_output: Path, csv_output: Path, failed_csv_output: Path, max_retries: int, request_delay: float, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> Tuple[int, int, int]:
    processed_files = 0
    flagged_files = 0
    all_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for rst_file in files:
        row = process_file(rst_file, workspace, source_root, client, max_retries, request_delay, max_output_tokens)
        if row is None:
            continue
        processed_files += 1
        json_row = build_json_row(row)
        all_rows.append(json_row)
        csv_rows.extend(build_csv_rows(row, workspace))
        has_flagged = (
            json_row["summary"]["partial"] > 0
            or json_row["summary"]["fail"] > 0
            or json_row["status"]["error"] is not None
            or any(ref.get("error") == "invalid image" for ref in row.image_refs)
        )
        if has_flagged:
            flagged_files += 1
        json_output.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(csv_output, csv_rows)
        write_csv(failed_csv_output, [csv_row for csv_row in csv_rows if csv_row.get("Result") == "Fail"])
    return processed_files, flagged_files, len(all_rows)


def enforce_strict_mode(json_output: Path) -> None:
    if not json_output.exists():
        return
    data = json.loads(json_output.read_text(encoding="utf-8"))
    for row in data:
        image_refs = row.get("image_refs", [])
        if any(ref.get("error") == "invalid image" for ref in image_refs):
            raise SystemExit(1)
        status = row.get("status") or {}
        if status.get("error") in {"invalid image", "backend_error"}:
            raise SystemExit(1)
        results = row.get("results")
        if not isinstance(results, list):
            raise SystemExit(1)
        for item in results:
            if not _is_complete_result_item(item):
                raise SystemExit(1)
            if item.get("hard_fail") is True or item.get("verdict") == "fail":
                raise SystemExit(1)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def human_duration(seconds: int) -> str:
    return f"{seconds // 60}m {seconds % 60}s"


def human_total_duration(seconds: int) -> str:
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m {seconds % 60}s"


def run_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, capture_output: bool = True, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=capture_output, check=check)


def load_bash_env(env_file: Path, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if not env_file.exists():
        raise SystemExit(f"Environment file not found: {env_file}")
    quoted = shlex.quote(str(env_file))
    proc = subprocess.run(["/bin/bash", "-c", f"set -a && source {quoted} && env -0"], text=False, capture_output=True, env=dict(base_env or os.environ))
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise SystemExit(f"Could not load environment file {env_file}: {stderr}")
    env: Dict[str, str] = {}
    for chunk in proc.stdout.split(b"\x00"):
        if chunk and b"=" in chunk:
            key, value = chunk.split(b"=", 1)
            env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return env


def list_repos_with_gh(org: str, limit: int, env: Dict[str, str]) -> List[str]:
    proc = run_command(["gh", "repo", "list", org, "--visibility=public", "--limit", str(limit), "--json", "nameWithOwner", "--jq", ".[].nameWithOwner"], env=env)
    if proc.returncode != 0:
        raise SystemExit(f"gh repo list failed:\n{proc.stderr}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def append_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def clone_or_pull_repo(full_repo: str, working_dir: Path, env: Dict[str, str]) -> Tuple[bool, Optional[str], str]:
    if (working_dir / ".git").is_dir():
        proc = run_command(["git", "-C", str(working_dir), "pull", "--ff-only"], env=env)
        error_message = "git pull failed"
    else:
        proc = run_command(["git", "clone", f"https://github.com/{full_repo}.git", str(working_dir)], env=env)
        error_message = "clone failed"
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, None if proc.returncode == 0 else error_message, output


def run_single_workspace_mode(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger = logging.getLogger(__name__)
    if not args.api_url:
        raise SystemExit("Missing AI API URL. Use --api-url or set AI_API_URL.")
    if not args.api_key:
        raise SystemExit("Missing AI API key. Use --api-key or set AI_API_KEY.")
    workspace = Path(args.workspace).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    files = select_input_files(args, workspace)
    logger.info("Starting AssetGuard")
    logger.info("Workspace: %s", workspace)
    logger.info("Found RST files: %d", len(files))
    client = ResponsesClient(args.api_url, args.api_key, args.model)
    processed_files, flagged_files, row_count = process_files(files, workspace, source_root, client, Path(args.output_json), Path(args.output_csv), Path(args.output_failed_csv), args.max_retries, args.request_delay, args.max_output_tokens)
    if args.strict:
        enforce_strict_mode(Path(args.output_json))
    logger.info("Done. Processed %d RST files, wrote %d rows to %s, flagged %d files, csv=%s, failed_csv=%s", processed_files, row_count, args.output_json, flagged_files, args.output_csv, args.output_failed_csv)


def _repo_worker(worker_args: Dict[str, Any]) -> Dict[str, Any]:
    full_repo = worker_args["full_repo"]
    repo_name = full_repo.split("/", 1)[1]
    clone_base = Path(worker_args["clone_base"]).expanduser().resolve()
    result_base = Path(worker_args["result_base"]).expanduser().resolve()
    env = dict(worker_args["env"])
    working_dir = clone_base / repo_name
    source_root = working_dir / "umn" / "source"
    result_dir = result_base / repo_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_log = result_dir / "run.log"
    duration_file = result_dir / "duration_seconds.txt"
    started = int(time.time())
    log_lines = [f"==> Processing {repo_name}\n"]
    ok, git_error, git_output = clone_or_pull_repo(full_repo, working_dir, env)
    log_lines.append(git_output.strip() + "\n")
    if not ok or not source_root.is_dir():
        reason = git_error if not ok else f"source root missing ({source_root})"
        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")
        log_lines.extend([f"{repo_name}: {reason}\n", f"duration_seconds={duration}\n", f"duration_human={human_duration(duration)}\n"])
        run_log.write_text("".join(log_lines), encoding="utf-8")
        return {"repo_name": repo_name, "success": False, "failure_reason": reason, "duration_seconds": duration, "duration_human": human_duration(duration), "result_dir": str(result_dir)}
    try:
        client = ResponsesClient(worker_args["api_url"], worker_args["api_key"], worker_args["model"])
        processed_files, flagged_files, row_count = process_files(find_rst_files(working_dir, []), working_dir, source_root, client, result_dir / "results_with_images.json", result_dir / "results_with_images.csv", result_dir / "results_with_images.failed_only.csv", worker_args["max_retries"], worker_args["request_delay"], worker_args["max_output_tokens"])
        if worker_args["strict"]:
            enforce_strict_mode(result_dir / "results_with_images.json")
        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")
        log_lines.extend([f"OK: stored results for {repo_name} in {result_dir} ({human_duration(duration)})\n", f"processed_files={processed_files}\n", f"flagged_files={flagged_files}\n", f"row_count={row_count}\n"])
        run_log.write_text("".join(log_lines), encoding="utf-8")
        return {"repo_name": repo_name, "success": True, "failure_reason": None, "duration_seconds": duration, "duration_human": human_duration(duration), "result_dir": str(result_dir)}
    except Exception as exc:
        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")
        log_lines.extend([f"ERROR: Processing failed for {repo_name}\n", f"exception={exc}\n"])
        run_log.write_text("".join(log_lines), encoding="utf-8")
        return {"repo_name": repo_name, "success": False, "failure_reason": str(exc), "duration_seconds": duration, "duration_human": human_duration(duration), "result_dir": str(result_dir)}


def run_full_repo_test(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger = logging.getLogger(__name__)
    clone_base = Path(args.clone_base).expanduser().resolve()
    result_base = Path(args.result_base).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    clone_base.mkdir(parents=True, exist_ok=True)
    result_base.mkdir(parents=True, exist_ok=True)
    failed_file = result_base / "failed_repos.txt"
    failed_file.write_text("", encoding="utf-8")
    env = load_bash_env(env_file, os.environ)
    env.setdefault("HOME", os.environ.get("HOME", str(Path.home())))
    args.api_url = args.api_url or env.get("AI_API_URL") or os.getenv("AI_API_URL")
    args.api_key = args.api_key or env.get("AI_API_KEY") or os.getenv("AI_API_KEY")
    args.model = args.model or env.get("AI_MODEL") or os.getenv("AI_MODEL", "qwen3.6-35b")
    if not args.api_url or not args.api_key:
        raise SystemExit("Missing AI API URL or API key.")
    repos = list_repos_with_gh(args.org, args.repo_limit, env)
    logger.info("Starting processing for %d repositories", len(repos))
    payloads = [{"full_repo": repo, "clone_base": str(clone_base), "result_base": str(result_base), "env": env, "api_url": args.api_url, "api_key": args.api_key, "model": args.model, "max_retries": args.max_retries, "request_delay": args.request_delay, "max_output_tokens": args.max_output_tokens, "strict": args.strict} for repo in repos]
    total_duration = 0
    success_count = 0
    failure_count = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for index, payload in enumerate(payloads):
            if index > 0 and args.worker_start_delay > 0:
                time.sleep(args.worker_start_delay)
            futures.append(executor.submit(_repo_worker, payload))
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            total_duration += int(result.get("duration_seconds") or 0)
            if result["success"]:
                success_count += 1
                logger.info("[%d/%d | OK=%d | FAILED=%d] OK: %s completed in %s", completed_count, len(futures), success_count, failure_count, result["repo_name"], result["duration_human"])
            else:
                failure_count += 1
                logger.error("[%d/%d | OK=%d | FAILED=%d] FAILED: %s: %s", completed_count, len(futures), success_count, failure_count, result["repo_name"], result["failure_reason"])
                append_text(failed_file, f"{result['repo_name']}: {result['failure_reason']}\n")
    logger.info("Finished. Repositories=%d, OK=%d, FAILED=%d, total duration=%s", len(repos), success_count, failure_count, human_total_duration(total_duration))


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
    parser.add_argument("--output-csv", default="results_with_images.csv", help="Concise CSV output with findings only.")
    parser.add_argument("--output-failed-csv", default="results_with_images.failed_only.csv", help="CSV output containing failed findings only.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 when score < 0.55, hard fail, backend error, invalid image, or invalid parsed JSON.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. DEBUG, INFO, WARNING.")
    parser.add_argument("--full-repo-test", action="store_true", help="Run the full multi-repository test workflow.")
    parser.add_argument("--org", default=DEFAULT_ORG, help="GitHub organization name for --full-repo-test.")
    parser.add_argument("--repo-limit", type=int, default=DEFAULT_REPO_LIMIT, help="Maximum number of repositories to fetch for --full-repo-test.")
    parser.add_argument("--clone-base", default="~/repotesting", help="Clone directory base for --full-repo-test.")
    parser.add_argument("--script-base", default="~/AssetGuard", help="Script base directory for --full-repo-test.")
    parser.add_argument("--result-base", default="~/AssetGuard/repo_results", help="Result directory base for --full-repo-test.")
    parser.add_argument("--env-file", default=".rst_checker__env", help="Bash environment file to source for --full-repo-test.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers for --full-repo-test.")
    parser.add_argument("--worker-start-delay", type=float, default=DEFAULT_WORKER_START_DELAY, help="Delay in seconds between scheduling worker starts for --full-repo-test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_repo_test:
        run_full_repo_test(args)
    else:
        run_single_workspace_mode(args)


if __name__ == "__main__":
    main()
