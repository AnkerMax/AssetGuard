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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from json import JSONDecoder, JSONDecodeError
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
REQUEST_READ_TIMEOUT = 240
DEFAULT_MAX_RETRIES = 2
DEFAULT_REQUEST_DELAY = 1
DEFAULT_MAX_OUTPUT_TOKENS = 8000
MAX_RESPONSE_EXCERPT_CHARS = 4000

DEFAULT_ORG = "opentelekomcloud-docs"
DEFAULT_REPO_LIMIT = 105
DEFAULT_MAX_WORKERS = 8
DEFAULT_WORKER_START_DELAY = 0.5

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
                            "image_kind": {
                                "type": "string",
                                "enum": ["screenshot", "icon", "other"]
                            },
                            "contains_interactive_buttons": {"type": "boolean"},
                            "buttons_magenta": {"type": "boolean"},
                            "hard_fail": {"type": "boolean"},
                            "hard_fail_reason": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"}
                                ]
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
    criteria = item.get("criteria", {})
    score = compute_overall_score(criteria)
    return verdict_from_score(score)


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
                refs.append(
                    ImageReference(
                        kind=kind,
                        name=match.group(1).strip(),
                        target=match.group(2).strip(),
                        line=idx,
                    )
                )
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
    suffix = path.suffix.lower()
    media_type = MEDIA_TYPES_BY_SUFFIX.get(suffix)
    if not media_type:
        raise ValueError("kein valides Bild")
    return media_type


def resolve_local_path(
    rst_file: Path,
    target: str,
    workspace: Path,
    source_root: Optional[Path] = None,
) -> Path:
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

    media_type = get_media_type_for_path(path)

    return LoadedImage(
        path=str(path.resolve()),
        media_type=media_type,
        data_base64=base64.b64encode(raw).decode("utf-8"),
    )


def build_image_candidates(
    rst_path: Path,
    refs: List[ImageReference],
    workspace: Path,
    source_root: Optional[Path] = None,
) -> List[ImageReference]:
    candidates: List[ImageReference] = []

    for ref in refs:
        try:
            resolved = resolve_local_path(rst_path, ref.target, workspace, source_root)
            valid = is_valid_image_path(ref.target)
            exists = resolved.exists()
            error = None
            if not valid or not exists:
                error = "kein valides Bild"

            candidates.append(
                ImageReference(
                    kind=ref.kind,
                    name=ref.name,
                    target=ref.target,
                    line=ref.line,
                    original_target=ref.target,
                    original_resolved_path=str(resolved),
                    resolved_path=str(resolved),
                    exists=exists,
                    is_valid_image=valid,
                    error=error,
                )
            )
        except ValueError:
            candidates.append(
                ImageReference(
                    kind=ref.kind,
                    name=ref.name,
                    target=ref.target,
                    line=ref.line,
                    original_target=ref.target,
                    original_resolved_path=None,
                    resolved_path=None,
                    exists=False,
                    is_valid_image=False,
                    error="kein valides Bild",
                )
            )

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
        "- If the image is a screenshot, check whether visible user-interactive buttons are magenta or plain white.\n"
        "- User-interactive buttons include clearly clickable UI controls such as buttons, CTA elements, or obvious interactive controls.\n"
        "- If the image is a screenshot and contains interactive buttons and those buttons are not magenta or plain white and has no other magenta buttons or text, set hard_fail=true.\n"
        "- In that case set hard_fail_reason to a short explanation.\n"
        "- In that case the image must be treated as failed regardless of the criteria scores.\n"
        "- If there are no visible interactive buttons, set contains_interactive_buttons=false and do not hard-fail for color.\n\n"
        "Scoring rules:\n"
        "- criteria scores still need to be filled for every image.\n"
        "- But if hard_fail=true, the final judgment must ignore the score.\n\n"
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
        "- hard_fail_reason must be null when hard_fail is false.\n"
        "- If no interactive buttons are visible, set buttons_magenta=false.\n"
        "- Do not add markdown fences.\n"
        "- Do not add analysis text before or after the JSON.\n\n"
        "Evidence rules:\n"
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


def extract_finish_reason(data: Dict[str, Any]) -> Optional[str]:
    if isinstance(data.get("status"), str):
        return data["status"]
    for item in data.get("output", []):
        if isinstance(item, dict) and item.get("finish_reason"):
            return item.get("finish_reason")
    return None


def _is_complete_result_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False

    required_top = {
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

    required_criteria = {
        "topic_match",
        "detail_match",
        "section_relevance",
        "visual_evidence",
        "contradictions",
    }
    if not required_criteria.issubset(criteria.keys()):
        return False

    for key in required_criteria:
        value = criteria.get(key)
        if not isinstance(value, int) or value < 0 or value > 3:
            return False

    if not isinstance(item.get("reasons"), list) or not all(isinstance(x, str) for x in item.get("reasons")):
        return False

    if not isinstance(item.get("missing_evidence"), list) or not all(
        isinstance(x, str) for x in item.get("missing_evidence")
    ):
        return False

    return True


def _normalize_candidate(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"results": obj}
    return None


def _extract_json_candidates_from_text(text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    decoder = JSONDecoder()
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue

        try:
            obj, end = decoder.raw_decode(text, i)
            normalized = _normalize_candidate(obj)
            if normalized is not None:
                candidates.append(normalized)
            i = max(i + 1, end)
        except JSONDecodeError:
            i += 1

    return candidates


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
                if (
                    isinstance(part, dict)
                    and part.get("type") in {"output_text", "text"}
                    and isinstance(part.get("text"), str)
                ):
                    txt = part["text"].strip()
                    if txt:
                        parts.append(txt)

    seen = set()
    deduped = []
    for p in parts:
        key = p[:500]
        if key not in seen:
            seen.add(key)
            deduped.append(p)

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
        i = 0
        while i < len(text):
            if text[i] not in "{[":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(text, i)
                add_candidate(obj)
                i = max(i + 1, end)
            except JSONDecodeError:
                i += 1

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
                ptype = part.get("type")
                if ptype in {"output_json", "json"}:
                    add_candidate(part.get("json"))
                elif ptype in {"output_text", "text"} and isinstance(part.get("text"), str):
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
        if isinstance(results, list) and len(results) > 0 and all(_is_complete_result_item(x) for x in results):
            return candidate

    return None


class ResponsesClient:
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def build_payload(
        self,
        prompt: str,
        images: List[LoadedImage],
        max_output_tokens: int,
    ) -> Tuple[Dict[str, Any], List[str]]:
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

    def post_with_retries(
        self,
        payload: Dict[str, Any],
        attached_images: List[str],
        timeout: int = 180,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_delay: float = DEFAULT_REQUEST_DELAY,
    ) -> ApiResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_result: Optional[ApiResult] = None

        for attempt in range(1, max_retries + 1):
            if request_delay > 0:
                time.sleep(request_delay)

            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(REQUEST_CONNECT_TIMEOUT, min(timeout, REQUEST_READ_TIMEOUT)),
                )
                status_code = response.status_code
                response_text = response.text

                if status_code in {429, 500, 502, 503, 504}:
                    last_result = ApiResult(
                        raw_text="",
                        parsed_json=None,
                        attached_image_count=len(attached_images),
                        attached_images=attached_images,
                        raw_response=None,
                        http_status=status_code,
                        http_response_text=response_text,
                        finish_reason=None,
                        attempt=attempt,
                        max_retries=max_retries,
                        error="backend_error",
                    )
                    if attempt < max_retries:
                        continue
                    return last_result

                response.raise_for_status()

                try:
                    data = response.json()
                except Exception:
                    data = {"_non_json_response_text": response_text}

                parsed_json = extract_response_json(data) if isinstance(data, dict) else None
                raw_text = extract_response_text(data) if isinstance(data, dict) else ""

                return ApiResult(
                    raw_text=raw_text,
                    parsed_json=parsed_json,
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
                last_result = ApiResult(
                    raw_text="",
                    parsed_json=None,
                    attached_image_count=len(attached_images),
                    attached_images=attached_images,
                    raw_response=None,
                    http_status=None,
                    http_response_text=str(exc),
                    finish_reason=None,
                    attempt=attempt,
                    max_retries=max_retries,
                    error="backend_error",
                )
                if attempt < max_retries:
                    continue
                return last_result

            except requests.RequestException as exc:
                return ApiResult(
                    raw_text="",
                    parsed_json=None,
                    attached_image_count=len(attached_images),
                    attached_images=attached_images,
                    raw_response=None,
                    http_status=getattr(getattr(exc, "response", None), "status_code", None),
                    http_response_text=getattr(getattr(exc, "response", None), "text", str(exc)),
                    finish_reason=None,
                    attempt=attempt,
                    max_retries=max_retries,
                    error="backend_error",
                )

        return last_result or ApiResult(
            raw_text="",
            parsed_json=None,
            attached_image_count=len(attached_images),
            attached_images=attached_images,
            raw_response=None,
            http_status=None,
            http_response_text="",
            finish_reason=None,
            attempt=max_retries,
            max_retries=max_retries,
            error="backend_error",
        )

    def analyze_images(
        self,
        prompt: str,
        images: List[LoadedImage],
        timeout: int = 180,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        request_delay: float = DEFAULT_REQUEST_DELAY,
    ) -> ApiResult:
        payload, attached_images = self.build_payload(prompt, images, max_output_tokens)
        return self.post_with_retries(
            payload=payload,
            attached_images=attached_images,
            timeout=timeout,
            max_retries=max_retries,
            request_delay=request_delay,
        )


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


def make_row(
    rst_file: Path,
    workspace: Path,
    title: Optional[str],
    image_refs: List[ImageReference],
    result: ApiResult,
) -> AuditRow:
    rel_path = rst_file.relative_to(workspace).as_posix() if rst_file.is_relative_to(workspace) else str(rst_file)
    return AuditRow(
        file_path=rel_path,
        title=title,
        image_count=len(image_refs),
        image_refs=[
            {
                "original_target": img.original_target,
                "original_path": img.original_resolved_path,
                "used_path": img.resolved_path,
                "kind": img.kind,
                "line": img.line,
                "exists": img.exists,
                "is_valid_image": img.is_valid_image,
                "error": img.error,
            }
            for img in image_refs
        ],
        result=asdict(result),
    )


def process_file(
    rst_file: Path,
    workspace: Path,
    source_root: Optional[Path],
    client: ResponsesClient,
    max_retries: int,
    request_delay: float,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Optional[AuditRow]:
    if not rst_file.exists() or not rst_file.is_file():
        return None

    rst_raw = rst_file.read_text(encoding="utf-8", errors="replace")
    refs = extract_image_refs(rst_raw)
    image_refs = build_image_candidates(rst_file, refs, workspace, source_root)
    if not image_refs:
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
    job = {
        "file_path": rel_path,
        "title": extract_title(rst_raw),
        "rst_raw": rst_raw,
        "image_refs": image_refs,
    }
    prompt = make_prompt(job)

    if not loaded_images:
        result = ApiResult(
            raw_text="",
            parsed_json=None,
            attached_image_count=0,
            attached_images=[],
            raw_response=None,
            http_status=None,
            http_response_text="kein valides Bild",
            finish_reason=None,
            attempt=0,
            max_retries=max_retries,
            error="kein valides Bild",
        )
        return make_row(rst_file, workspace, job["title"], image_refs, result)

    try:
        result = client.analyze_images(
            prompt=prompt,
            images=loaded_images,
            max_retries=max_retries,
            request_delay=request_delay,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        result = ApiResult(
            raw_text="",
            parsed_json=None,
            attached_image_count=len(loaded_images),
            attached_images=[img.path for img in loaded_images],
            raw_response=None,
            http_status=None,
            http_response_text=str(exc),
            finish_reason=None,
            attempt=max_retries,
            max_retries=max_retries,
            error="backend_error",
        )

    return make_row(rst_file, workspace, job["title"], image_refs, result)


def build_json_row(row: AuditRow) -> Dict[str, Any]:
    parsed = ((row.result or {}).get("parsed_json")) or {}
    parsed_results = parsed.get("results", []) if isinstance(parsed, dict) else []

    enriched_results = []
    summary = {"pass": 0, "partial": 0, "fail": 0}

    for item in parsed_results:
        criteria = item.get("criteria", {})
        score = compute_overall_score(criteria)
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


def build_csv_rows(row: AuditRow) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    parsed = ((row.result or {}).get("parsed_json")) or {}
    parsed_results = parsed.get("results", []) if isinstance(parsed, dict) else []

    ref_map = {
        ref.get("used_path"): ref
        for ref in row.image_refs
        if ref.get("used_path")
    }

    for item in parsed_results:
        criteria = item.get("criteria", {})
        score = compute_overall_score(criteria)
        verdict = final_verdict(item)
        ref = ref_map.get(item.get("image_path"), {})

        rows.append({
            "document_file": row.file_path,
            "document_title": row.title or "",
            "image_file": item.get("image_path", ""),
            "image_reference_type": ref.get("kind", ""),
            "image_reference_line": ref.get("line", ""),
            "detected_image_type": item.get("image_kind", ""),
            "has_interactive_buttons": item.get("contains_interactive_buttons", ""),
            "interactive_buttons_magenta": item.get("buttons_magenta", ""),
            "hard_fail_triggered": item.get("hard_fail", ""),
            "hard_fail_reason": item.get("hard_fail_reason", "") or "",
            "score_topic_match": criteria.get("topic_match", ""),
            "score_detail_match": criteria.get("detail_match", ""),
            "score_section_relevance": criteria.get("section_relevance", ""),
            "score_visual_evidence": criteria.get("visual_evidence", ""),
            "score_contradictions": criteria.get("contradictions", ""),
            "overall_score": f"{score:.2f}",
            "final_verdict": verdict,
            "processing_error": row.result.get("error", "") or "",
            "api_http_status": row.result.get("http_status", ""),
            "api_finish_reason": row.result.get("finish_reason", "") or "",
            "api_attempt": row.result.get("attempt", ""),
            "match_reasons": " | ".join(item.get("reasons", [])),
            "missing_evidence": " | ".join(item.get("missing_evidence", [])),
        })

    if not rows and row.result.get("error"):
        rows.append({
            "document_file": row.file_path,
            "document_title": row.title or "",
            "image_file": "",
            "image_reference_type": "",
            "image_reference_line": "",
            "detected_image_type": "",
            "has_interactive_buttons": "",
            "interactive_buttons_magenta": "",
            "hard_fail_triggered": "",
            "hard_fail_reason": "",
            "score_topic_match": "",
            "score_detail_match": "",
            "score_section_relevance": "",
            "score_visual_evidence": "",
            "score_contradictions": "",
            "overall_score": "",
            "final_verdict": "fail",
            "processing_error": row.result.get("error", "") or "",
            "api_http_status": row.result.get("http_status", ""),
            "api_finish_reason": row.result.get("finish_reason", "") or "",
            "api_attempt": row.result.get("attempt", ""),
            "match_reasons": "",
            "missing_evidence": "",
        })

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "document_file",
        "document_title",
        "image_file",
        "image_reference_type",
        "image_reference_line",
        "detected_image_type",
        "has_interactive_buttons",
        "interactive_buttons_magenta",
        "hard_fail_triggered",
        "hard_fail_reason",
        "score_topic_match",
        "score_detail_match",
        "score_section_relevance",
        "score_visual_evidence",
        "score_contradictions",
        "overall_score",
        "final_verdict",
        "processing_error",
        "api_http_status",
        "api_finish_reason",
        "api_attempt",
        "match_reasons",
        "missing_evidence",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_files(
    files: List[Path],
    workspace: Path,
    source_root: Optional[Path],
    client: ResponsesClient,
    json_output: Path,
    csv_output: Path,
    failed_csv_output: Path,
    max_retries: int,
    request_delay: float,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Tuple[int, int, int]:
    processed_files = 0
    flagged_files = 0
    all_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for rst_file in files:
        row = process_file(
            rst_file=rst_file,
            workspace=workspace,
            source_root=source_root,
            client=client,
            max_retries=max_retries,
            request_delay=request_delay,
            max_output_tokens=max_output_tokens,
        )
        if row is None:
            continue

        processed_files += 1

        json_row = build_json_row(row)
        all_rows.append(json_row)

        file_csv_rows = build_csv_rows(row)
        csv_rows.extend(file_csv_rows)

        has_flagged = (
            json_row["summary"]["partial"] > 0
            or json_row["summary"]["fail"] > 0
            or json_row["status"]["error"] is not None
            or any(ref.get("error") == "kein valides Bild" for ref in row.image_refs)
        )
        if has_flagged:
            flagged_files += 1

    json_output.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_output, csv_rows)
    write_csv(failed_csv_output, [row for row in csv_rows if row.get("final_verdict") == "fail"])

    return processed_files, flagged_files, len(all_rows)


def enforce_strict_mode(json_output: Path) -> None:
    if not json_output.exists():
        return

    data = json.loads(json_output.read_text(encoding="utf-8"))
    for row in data:
        image_refs = row.get("image_refs", [])
        if any(ref.get("error") == "kein valides Bild" for ref in image_refs):
            raise SystemExit(1)

        status = row.get("status") or {}
        if status.get("error") in {"kein valides Bild", "backend_error"}:
            raise SystemExit(1)

        results = row.get("results")
        if not isinstance(results, list):
            raise SystemExit(1)

        for item in results:
            if not _is_complete_result_item(item):
                raise SystemExit(1)
            if item.get("hard_fail") is True:
                raise SystemExit(1)
            if item.get("verdict") == "fail":
                raise SystemExit(1)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def human_duration(seconds: int) -> str:
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes}m {sec}s"


def human_total_duration(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return f"{hours}h {minutes}m {sec}s"


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def load_bash_env(env_file: Path, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if not env_file.exists():
        raise SystemExit(f"Env-Datei nicht gefunden: {env_file}")

    seed_env = dict(base_env or os.environ)
    quoted = shlex.quote(str(env_file))
    cmd = f"set -a && source {quoted} && env -0"

    proc = subprocess.run(
        ["/bin/bash", "-c", cmd],
        text=False,
        capture_output=True,
        env=seed_env,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise SystemExit(f"Fehler beim Laden der Env-Datei {env_file}: {stderr}")

    env: Dict[str, str] = {}
    for chunk in proc.stdout.split(b"\x00"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")

    return env


def list_repos_with_gh(org: str, limit: int, env: Dict[str, str]) -> List[str]:
    proc = run_command(
        [
            "gh", "repo", "list", org,
            "--visibility=public",
            "--limit", str(limit),
            "--json", "nameWithOwner",
            "--jq", ".[].nameWithOwner",
        ],
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(f"gh repo list fehlgeschlagen:\n{proc.stderr}")

    repos = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return repos


def append_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def clone_or_pull_repo(
    full_repo: str,
    working_dir: Path,
    env: Dict[str, str],
) -> Tuple[bool, Optional[str], str]:
    if (working_dir / ".git").is_dir():
        proc = run_command(["git", "-C", str(working_dir), "pull", "--ff-only"], env=env)
        if proc.returncode != 0:
            return False, "git pull fehlgeschlagen", (proc.stdout or "") + "\n" + (proc.stderr or "")
        return True, None, (proc.stdout or "") + "\n" + (proc.stderr or "")
    else:
        proc = run_command(["git", "clone", f"https://github.com/{full_repo}.git", str(working_dir)], env=env)
        if proc.returncode != 0:
            return False, "clone fehlgeschlagen", (proc.stdout or "") + "\n" + (proc.stderr or "")
        return True, None, (proc.stdout or "") + "\n" + (proc.stderr or "")


def run_single_workspace_mode(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
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
    logger.info("Found RST-Dateien: %d", len(files))
    logger.info("Sending files to backend and waiting for response..")

    client = ResponsesClient(api_url=args.api_url, api_key=args.api_key, model=args.model)

    processed_files, flagged_files, row_count = process_files(
        files=files,
        workspace=workspace,
        source_root=source_root,
        client=client,
        json_output=Path(args.output_json),
        csv_output=Path(args.output_csv),
        failed_csv_output=Path(args.output_failed_csv),
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        max_output_tokens=args.max_output_tokens,
    )

    if args.strict:
        enforce_strict_mode(Path(args.output_json))

    logger.info(
        "Done. Processed %d rst files, wrote %d rows to %s, flagged %d files, csv=%s, failed_csv=%s",
        processed_files,
        row_count,
        args.output_json,
        flagged_files,
        args.output_csv,
        args.output_failed_csv,
    )


def _repo_worker(worker_args: Dict[str, Any]) -> Dict[str, Any]:
    repo_logger = logging.getLogger(__name__)

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
    output_json = result_dir / "results_with_images.json"
    output_csv = result_dir / "results_with_images.csv"
    output_failed_csv = result_dir / "results_with_images.failed_only.csv"

    started = int(time.time())

    log_lines = [f"==> Bearbeite {repo_name}\n"]

    ok, git_error, git_output = clone_or_pull_repo(full_repo, working_dir, env)
    log_lines.append(git_output.strip() + "\n")

    if not ok:
        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")
        log_lines.append(f"{repo_name}: {git_error}\n")
        log_lines.append(f"duration_seconds={duration}\n")
        log_lines.append(f"duration_human={human_duration(duration)}\n")
        run_log.write_text("".join(log_lines), encoding="utf-8")
        return {
            "repo_name": repo_name,
            "success": False,
            "failure_reason": git_error,
            "duration_seconds": duration,
            "duration_human": human_duration(duration),
            "result_dir": str(result_dir),
        }

    if not source_root.is_dir():
        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")
        msg = f"source root fehlt ({source_root})"
        log_lines.append(f"{repo_name}: {msg}\n")
        log_lines.append(f"duration_seconds={duration}\n")
        log_lines.append(f"duration_human={human_duration(duration)}\n")
        run_log.write_text("".join(log_lines), encoding="utf-8")
        return {
            "repo_name": repo_name,
            "success": False,
            "failure_reason": msg,
            "duration_seconds": duration,
            "duration_human": human_duration(duration),
            "result_dir": str(result_dir),
        }

    try:
        client = ResponsesClient(
            api_url=worker_args["api_url"],
            api_key=worker_args["api_key"],
            model=worker_args["model"],
        )

        files = find_rst_files(working_dir, [])
        processed_files, flagged_files, row_count = process_files(
            files=files,
            workspace=working_dir,
            source_root=source_root,
            client=client,
            json_output=output_json,
            csv_output=output_csv,
            failed_csv_output=output_failed_csv,
            max_retries=worker_args["max_retries"],
            request_delay=worker_args["request_delay"],
            max_output_tokens=worker_args["max_output_tokens"],
        )

        if worker_args["strict"]:
            enforce_strict_mode(output_json)

        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")

        log_lines.append(
            f"OK: speichere Ergebnis von {repo_name} in {result_dir} "
            f"(Dauer: {human_duration(duration)})\n"
        )
        log_lines.append(f"processed_files={processed_files}\n")
        log_lines.append(f"flagged_files={flagged_files}\n")
        log_lines.append(f"row_count={row_count}\n")
        log_lines.append(f"duration_seconds={duration}\n")
        log_lines.append(f"duration_human={human_duration(duration)}\n")
        run_log.write_text("".join(log_lines), encoding="utf-8")

        return {
            "repo_name": repo_name,
            "success": True,
            "failure_reason": None,
            "duration_seconds": duration,
            "duration_human": human_duration(duration),
            "result_dir": str(result_dir),
        }

    except Exception as exc:
        duration = int(time.time()) - started
        duration_file.write_text(str(duration), encoding="utf-8")
        log_lines.append(
            f"FEHLER: Python-Skript für {repo_name} fehlgeschlagen "
            f"(Dauer: {human_duration(duration)})\n"
        )
        log_lines.append(f"exception={exc}\n")
        log_lines.append(f"duration_seconds={duration}\n")
        log_lines.append(f"duration_human={human_duration(duration)}\n")
        run_log.write_text("".join(log_lines), encoding="utf-8")

        return {
            "repo_name": repo_name,
            "success": False,
            "failure_reason": str(exc),
            "duration_seconds": duration,
            "duration_human": human_duration(duration),
            "result_dir": str(result_dir),
        }


def run_full_repo_test(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    clone_base = Path(args.clone_base).expanduser().resolve()
    script_base = Path(args.script_base).expanduser().resolve()
    result_base = Path(args.result_base).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()

    clone_base.mkdir(parents=True, exist_ok=True)
    result_base.mkdir(parents=True, exist_ok=True)

    failed_file = result_base / "failed_repos.txt"
    failed_file.write_text("", encoding="utf-8")

    env = load_bash_env(env_file, os.environ)
    env.setdefault("HOME", os.environ.get("HOME", str(Path.home())))

    if not args.api_url:
        args.api_url = env.get("AI_API_URL") or os.getenv("AI_API_URL")
    if not args.api_key:
        args.api_key = env.get("AI_API_KEY") or os.getenv("AI_API_KEY")
    if not args.model:
        args.model = env.get("AI_MODEL") or os.getenv("AI_MODEL", "qwen3.6-35b")

    if not args.api_url:
        raise SystemExit("Missing AI API URL. Use --api-url or set AI_API_URL.")
    if not args.api_key:
        raise SystemExit("Missing AI API key. Use --api-key or set AI_API_KEY.")

    logger.info("Full repo test gestartet")
    logger.info("Org: %s", args.org)
    logger.info("Clone base: %s", clone_base)
    logger.info("Result base: %s", result_base)
    logger.info("Env file: %s", env_file)
    logger.info("Workers: %d", args.max_workers)
    logger.info("Worker start delay: %.2fs", args.worker_start_delay)

    repos = list_repos_with_gh(args.org, args.repo_limit, env)
    logger.info("Gefundene Repos: %d", len(repos))
    logger.info("Starte Verarbeitung von %d Repos", len(repos))

    worker_payloads: List[Dict[str, Any]] = []
    for full_repo in repos:
        worker_payloads.append({
            "full_repo": full_repo,
            "clone_base": str(clone_base),
            "script_base": str(script_base),
            "result_base": str(result_base),
            "env": env,
            "api_url": args.api_url,
            "api_key": args.api_key,
            "model": args.model,
            "max_retries": args.max_retries,
            "request_delay": args.request_delay,
            "max_output_tokens": args.max_output_tokens,
            "strict": args.strict,
        })

    total_duration = 0
    futures = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, payload in enumerate(worker_payloads):
            if idx > 0 and args.worker_start_delay > 0:
                time.sleep(args.worker_start_delay)
            futures.append(executor.submit(_repo_worker, payload))

        total_repos = len(futures)
        success_count = 0
        failure_count = 0

        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            repo_name = result["repo_name"]
            duration = int(result.get("duration_seconds") or 0)
            total_duration += duration
            percent = (completed_count / total_repos) * 100

            if result["success"]:
                success_count += 1
                logger.info(
                    "[%d/%d | %.1f%% | OK=%d | FEHLER=%d] OK: %s abgeschlossen in %s",
                    completed_count,
                    total_repos,
                    percent,
                    success_count,
                    failure_count,
                    repo_name,
                    result["duration_human"],
                )
            else:
                failure_count += 1
                logger.error(
                    "[%d/%d | %.1f%% | OK=%d | FEHLER=%d] FEHLER: %s fehlgeschlagen in %s: %s",
                    completed_count,
                    total_repos,
                    percent,
                    success_count,
                    failure_count,
                    repo_name,
                    result["duration_human"],
                    result["failure_reason"],
                )
                append_text(failed_file, f"{repo_name}: {result['failure_reason']}\n")
                append_text(failed_file, f"{repo_name} duration_seconds={duration}\n")
                append_text(failed_file, f"{repo_name} duration_human={result['duration_human']}\n")

    total_human = human_total_duration(total_duration)
    logger.info(
        "Fertig. Repos gesamt=%d, OK=%d, FEHLER=%d, Gesamtlaufzeit aller Repos: %s",
        len(repos),
        success_count,
        failure_count,
        total_human,
    )
    append_text(failed_file, f"Gesamtlaufzeit aller Repos: {total_human}\n")


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
    parser.add_argument("--output-failed-csv", default="results_with_images.failed_only.csv", help="Only failed rows as CSV.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 when score < 0.55, hard fail, backend error, invalid image, or invalid parsed JSON.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. DEBUG, INFO, WARNING.")

    parser.add_argument("--full-repo-test", action="store_true", help="Run the full multi-repo test workflow and replace the old bash orchestration.")
    parser.add_argument("--org", default=DEFAULT_ORG, help="GitHub organization name for --full-repo-test.")
    parser.add_argument("--repo-limit", type=int, default=DEFAULT_REPO_LIMIT, help="Maximum number of repos to fetch for --full-repo-test.")
    parser.add_argument("--clone-base", default="~/repotesting", help="Clone directory base for --full-repo-test.")
    parser.add_argument("--script-base", default="~/AssetGuard", help="Script base directory for --full-repo-test.")
    parser.add_argument("--result-base", default="~/AssetGuard/repo_results", help="Result directory base for --full-repo-test.")
    parser.add_argument("--env-file", default=".rst_checker__env", help="Bash env file to source for --full-repo-test.")
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
