# AssetGuard

AssetGuard is a Python CLI tool for auditing image references in reStructuredText (`.rst`) documentation. It extracts local image references from `.rst` files, resolves them against the repository, sends the document content plus the referenced images to a multimodal Responses API, and writes structured evaluation results to JSON and CSV outputs.

The tool is designed for documentation QA workflows where images should match the surrounding text context and optionally enforce strict failure conditions in CI.

## Features

- Extracts local image references from `.rst` files:
  - `.. image::`
  - `.. figure::`
  - substitution images such as `|name| image::`
- Resolves relative and leading-slash image paths
- Validates and loads supported local image files
- Sends `.rst` content and attached images to a multimodal API using a strict JSON schema
- Computes weighted scores and final verdicts per image
- Writes:
  - a structured JSON report per processed `.rst` file
  - a flat CSV report per evaluated image
  - a fail-only CSV for pipeline usage
- Supports strict mode for CI/CD

## Requirements

- Python 3.10 or newer recommended
- `pip`
- A Responses API-compatible multimodal endpoint
- A valid API key
- A model that supports image input and structured JSON output
- A workspace containing `.rst` files
- Optional: a separate `source root` for leading-slash image paths

## Installation

Install the required dependency:

```bash
pip install requests
```

## Configuration

AssetGuard reads these environment variables:

- `AI_API_URL`
- `AI_API_KEY`
- `AI_MODEL`

Example `.env` file:

```env
AI_API_URL=https://your-endpoint.example/v1/responses
AI_API_KEY=your_api_key_example
AI_MODEL=your_model_name
```

The script does **not** load `.env` automatically. Load it before execution:

```bash
set -a
source .env
set +a
```

You can also pass these values via CLI arguments.

## Quick start

Run on the current workspace:

```bash
python3 assetguard.py --workspace . --source-root /path/to/source/root
```

Run on a specific workspace:

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root
```

After the run, inspect the output files:

```bash
cat results_with_images.json
```

```bash
cat results_with_images.csv
```

```bash
cat results_with_images.failed_only.csv
```

## How it works

For each `.rst` file, AssetGuard:

1. Reads the file content
2. Extracts image references from supported RST directives
3. Resolves local image paths
4. Filters to supported image types
5. Loads valid local images and base64-encodes them
6. Sends the `.rst` content and images to the API
7. Parses the structured model response
8. Computes an overall weighted score per image
9. Assigns a final verdict (`pass`, `partial`, or `fail`)
10. Writes aggregated file-level JSON and image-level CSV outputs

Files without image references are skipped.

## Supported image types

Currently supported image suffixes:

- `.png`
- `.jpg`
- `.jpeg`

Other image types such as `.webp` and `.gif` are **not** supported in the current code.

## Scoring and verdicts

Each image is scored on these criteria:

- `topic_match`
- `detail_match`
- `section_relevance`
- `visual_evidence`
- `contradictions`

Weights:

- `topic_match`: 0.30
- `detail_match`: 0.20
- `section_relevance`: 0.20
- `visual_evidence`: 0.15
- `contradictions`: 0.15

Overall score:

$\text{score} = \frac{\sum_k w_k \cdot c_k}{\sum_k 3 \cdot w_k}$ ; range [0,1] ; where $\(c_k\)$ is criterion score (0–3) and $\(w_k\)$ is corresponding weight

The score is rounded to two decimal places.

Verdict thresholds:

- `pass`: score >= 0.80
- `partial`: 0.55 <= score < 0.80
- `fail`: score < 0.55

## Hard-fail rule

In addition to normal scoring, the prompt applies a special hard-fail rule for screenshots:

- If an image is classified as a screenshot
- and it contains visible interactive buttons
- and those buttons are neither magenta nor plain white

then the model is instructed to set:

- `hard_fail = true`
- `hard_fail_reason = <short explanation>`

A hard fail always results in a final verdict of `fail`, regardless of the numeric score.

## Usage

### Process all `.rst` files in a workspace

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root
```

### Process one file

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --rst-file path/to/file.rst
```

### Process multiple files

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --rst-file path/to/file1.rst \
  --rst-file path/to/file2.rst
```

### Process files listed in a text file

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --file-list rst_files.txt
```

### Restrict processing to path prefixes

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --path-prefix umn/source/api_management \
  --path-prefix api-ref/source
```

### Pass API values directly

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --api-url "$AI_API_URL" \
  --api-key "$AI_API_KEY" \
  --model "$AI_MODEL"
```

### Enable strict mode

```bash
python3 assetguard.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --strict
```

## CLI arguments

| Argument | Description |
|---------|-------------|
| `--workspace` | Local repo/workspace path. Defaults to `.` |
| `--source-root` | Optional source root for leading-slash image paths |
| `--file-list` | Text file with one repo-relative `.rst` path per line |
| `--rst-file` | Single `.rst` file to process; can be repeated |
| `--path-prefix` | Restrict processing to `.rst` files whose relative path starts with this prefix |
| `--api-url` | Responses API endpoint |
| `--api-key` | API key |
| `--model` | Model name; defaults to `AI_MODEL` or `qwen3.6-35b` |
| `--request-delay` | Fixed delay before each API call |
| `--max-retries` | Retry count for transient/backend errors |
| `--max-output-tokens` | Maximum output tokens for the API response |
| `--output-json` | JSON output path, default `results_with_images.json` |
| `--output-csv` | CSV output path, default `results_with_images.csv` |
| `--output-failed-csv` | Fail-only CSV output path, default `results_with_images.failed_only.csv` |
| `--strict` | Exit with code 1 on fail conditions |
| `--log-level` | Logging level, e.g. `DEBUG`, `INFO`, `WARNING` |

## Output files

### `results_with_images.json`

File-level structured output. Each row contains:

- `file_path`
- `title`
- `image_count`
- `image_refs`
- `status`
- `summary`
- `results`

Notes:
- JSON contains **all processed `.rst` files with image references**, not only failed or partial ones.
- `summary` includes counts for `pass`, `partial`, and `fail`.
- `results` contains enriched per-image results with:
  - `overall_score`
  - `verdict`

### `results_with_images.csv`

Flat image-level CSV output. Includes:

- document path and title
- image path
- reference type and line
- detected image type
- button checks
- hard-fail status
- per-criterion scores
- overall score
- final verdict
- processing/API status fields
- reasons and missing evidence

### `results_with_images.failed_only.csv`

Contains only rows from the flat CSV where:

- `final_verdict == fail`

This file is useful for CI pipelines and focused review.

## Strict mode behavior

When `--strict` is enabled, AssetGuard exits with status code `1` if any processed row contains one of these conditions:

- invalid image reference (`kein valides Bild`)
- backend/API error (`backend_error`)
- invalid parsed result structure
- `hard_fail == true`
- final verdict `fail`

`partial` alone does **not** trigger strict mode failure unless another strict failure condition is present.

## Path resolution rules

AssetGuard resolves image paths as follows:

- Relative paths are resolved relative to the `.rst` file location
- Paths starting with `/` are resolved relative to:
  - `--source-root`, if provided
  - otherwise `--workspace`

Non-local targets such as:

- `http://...`
- `https://...`
- `data:...`

are treated as invalid local images in the current implementation.

## Notes

- Duplicate loaded image paths are deduplicated before submission
- The full `.rst` content is embedded into the model prompt
- Large `.rst` files may increase latency and token usage
- Files with no extracted image references are skipped entirely
- Missing or unsupported local images are marked as `kein valides Bild`
- Transient backend statuses `429`, `500`, `502`, `503`, and `504` are retried up to `--max-retries`
- Logging is configurable with `--log-level`

## Use cases

- Documentation image validation
- Technical documentation QA
- Detecting weak, misleading, or mismatched visuals
- CI/CD gating with `--strict`
- Producing machine-readable audit output for pipelines
- Establishing a base for future visual compliance checks
