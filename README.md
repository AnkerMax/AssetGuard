# AssetGuard

AssetGuard is a Python CLI tool for auditing local image references in reStructuredText (`.rst`) documentation. It finds supported image directives, resolves the referenced files, checks images locally, sends valid images together with the complete RST document to a multimodal API, evaluates the relationship between text and image, and writes machine-readable reports.

The tool can be used in two ways:

- **Single-workspace mode:** audit one local repository or documentation workspace.
- **Full-repo-test mode:** discover repositories in a GitHub organization, clone or update them, audit their RST files in parallel, and store separate results per repository.

The intended use case is documentation QA and CI/CD gating: images should be present, readable, relevant to the surrounding documentation, and free from configured local hard-fail conditions.

## Features

- Extracts local image references from `.rst` files:
  - `.. image::`
  - `.. figure::`
  - substitution images such as `|name| image::`
- Resolves relative paths and leading-slash paths.
- Supports `.png`, `.jpg`, and `.jpeg` files.
- Rejects remote references such as `http://`, `https://`, and `data:` URLs.
- Checks image files locally before sending them to the API.
- Detects the configured forbidden color `#3298ff` locally and creates a hard-fail result without sending that image to the API.
- Sends the complete RST content and valid images to a multimodal OpenAI-compatible Chat Completions endpoint.
- Requires structured JSON output from the model.
- Computes weighted scores and final verdicts per image.
- Writes a detailed JSON report, a compact user-facing CSV, and a debug CSV containing only cases that deserve investigation.
- Supports configurable retries, delays, logging, and strict CI mode.
- Supports parallel testing of multiple repositories through `--full-repo-test`.

## Requirements

### Single-workspace mode

- Python 3.10 or newer recommended.
- `requests`.
- `Pillow`.
- An OpenAI-compatible Chat Completions multimodal API endpoint.
- An API key.
- A model that supports image input and structured JSON output.
- A workspace containing `.rst` files.

### Full-repo-test mode

In addition to the requirements above:

- Git.
- GitHub CLI (`gh`), authenticated with permission to list and clone the relevant repositories.
- Bash, because the configured environment file is sourced through `/bin/bash`.
- A source root named `umn/source` inside each cloned repository. Repositories without this directory are marked as failed.

Install the Python dependencies with:

```bash
python3 -m pip install requests Pillow
```

## Configuration

The following environment variables are supported:

- `AI_API_URL` – API endpoint, for example `https://your-endpoint.example/v1/chat/completions`.
- `AI_API_KEY` – API key used for the bearer-authenticated request.
- `AI_MODEL` – model name. The default is `qwen3.6-35b` if no value is supplied.

Example environment file:

```env
AI_API_URL=https://your-endpoint.example/v1/chat/completions
AI_API_KEY=your_api_key_example
AI_MODEL=your_model_name
```

The normal single-workspace mode does not load a `.env` file automatically. Export the variables before starting the script or pass the values through CLI arguments:

```bash
set -a
source .env
set +a
```

For `--full-repo-test`, the file specified by `--env-file` is sourced by Bash. The default is `.rst_checker__env`. This mode can therefore load the API values from that file automatically.

Do not commit API keys to the repository.

## Quick start

### Audit the current workspace

```bash
export AI_API_URL="https://your-endpoint.example/v1/chat/completions"
export AI_API_KEY="your_api_key"
export AI_MODEL="your_model_name"

python3 AssetGuard_chat_completions.py --workspace .
```

The default output files are:

```text
results_with_images.json
results.csv
results_debug_failed.csv
```

### Audit one file

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --rst-file path/to/file.rst
```

### Run a full organization test

```bash
python3 AssetGuard_chat_completions.py \
  --full-repo-test \
  --org opentelekomcloud-docs \
  --env-file .rst_checker__env
```

The full-repo workflow lists public repositories with `gh`, clones new repositories or runs `git pull --ff-only` for existing clones, audits repositories in parallel, and stores exactly three result files per successfully processed repository below the configured result directory.

## How it works

For each selected `.rst` file, AssetGuard:

1. Reads the file as UTF-8, replacing undecodable characters if necessary.
2. Extracts supported image directives and records their line numbers.
3. Resolves image paths relative to the RST file or, for leading-slash paths, relative to `--source-root` or `--workspace`.
4. Marks missing, unsupported, and remote image references as `kein valides Bild`.
5. Opens valid local images with Pillow.
6. Checks valid images for the configured forbidden color `#3298ff`.
7. Excludes locally hard-failed images from the API request.
8. Builds one multimodal API request containing the RST text and the remaining unique images.
9. Parses the structured response and validates the required result fields.
10. Computes an overall score and verdict for every returned image.
11. Normalizes API results by image path, removes duplicate/unexpected model results, and records a warning when an attached image has no valid result.
12. Writes the detailed JSON report, compact result CSV, and investigation-only debug CSV.

Files without extracted image references are skipped and do not appear in the reports. Duplicate image paths are sent only once per document, although the original references remain recorded in the JSON metadata.

## Local image validation

The current implementation supports only these suffixes:

- `.png`
- `.jpg`
- `.jpeg`

The following are not supported as local image inputs:

- `.webp`
- `.gif`
- Remote HTTP(S) images.
- `data:` URLs.
- Missing files.

Unsupported or missing images are recorded with the error `kein valides Bild`. If all references in a document are invalid, the API is not called for that document.

### Forbidden-color check

Before an image is sent to the API, AssetGuard scans its RGB pixels for the exact color `#3298ff`. The configured tolerance is currently `0`, so only an exact RGB match triggers the check.

When the color is found:

- the image is not sent to the API;
- a local result with `hard_fail=true` is created;
- the final verdict is `fail`, regardless of its numeric score;
- the reason identifies the forbidden color.

This is a local pixel check. It is separate from the model's semantic evaluation and should be considered when changing image design rules.

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

The current implementation has two hard-fail paths:

1. **Local color hard fail:** an image contains the configured forbidden color `#3298ff`.
2. **Structured result hard fail:** a returned result contains `hard_fail=true`.

The current API prompt explicitly instructs the model not to use color as a hard-fail criterion, to set `hard_fail=false`, and to set `hard_fail_reason=null`. Therefore, color-based hard fails are expected to originate from the local check rather than from the model.

A hard fail always results in a final verdict of `fail`, regardless of the numeric score.

## Usage

### Process all `.rst` files

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root
```

Without `--rst-file`, `--file-list`, or `--path-prefix`, all `.rst` files below `--workspace` are selected recursively.

### Process selected files

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --rst-file path/to/file1.rst \
  --rst-file path/to/file2.rst
```

The `--rst-file` option can be repeated. Relative paths are interpreted relative to the workspace.

### Process a file list

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --file-list rst_files.txt
```

The file must contain one path per line. Empty lines and lines beginning with `#` are ignored.

### Restrict processing by path prefix

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --path-prefix umn/source/api_management \
  --path-prefix api-ref/source
```

A file is selected when its workspace-relative path starts with at least one supplied prefix.

### Pass API values directly

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --api-url "$AI_API_URL" \
  --api-key "$AI_API_KEY" \
  --model "$AI_MODEL"
```

### Enable strict mode

```bash
python3 AssetGuard_chat_completions.py \
  --workspace /path/to/workspace \
  --strict
```

Strict mode is intended for CI/CD. The command exits with status code `1` when a strict failure condition is found.

### Full-repo-test mode

```bash
python3 AssetGuard_chat_completions.py \
  --full-repo-test \
  --org opentelekom-cloud-docs \
  --repo-limit 105 \
  --clone-base ~/repotesting \
  --result-base ~/AssetGuard/repo_results \
  --env-file .rst_checker__env \
  --max-workers 8 \
  --worker-start-delay 0.5
```

In this mode:

- repositories are obtained with `gh repo list`;
- new repositories are cloned from GitHub;
- existing working copies are updated with `git pull --ff-only`;
- repositories are processed in parallel;
- each repository gets its own result directory;
- repository-level clone, source-root, strict-mode, and processing failures are reported through the process log/terminal;
- each successfully processed repository receives exactly three result files.

The script uses the organization default `opentelekomcloud-docs`, a repository limit of `105`, eight workers, and a worker start delay of `0.5` seconds unless overridden.

## CLI arguments

### General and single-workspace options

| Argument | Description |
|---|---|
| `--workspace` | Local repository or workspace path. Defaults to `.`. |
| `--source-root` | Source root used for leading-slash image paths. If omitted, `--workspace` is used. |
| `--file-list` | Text file containing one RST path per line. |
| `--rst-file` | RST file to process. Can be repeated. |
| `--path-prefix` | Restricts automatic discovery to paths beginning with this prefix. Can be repeated. |
| `--api-url` | API endpoint. Defaults to `AI_API_URL`. |
| `--api-key` | API key. Defaults to `AI_API_KEY`. |
| `--model` | Model name. Defaults to `AI_MODEL` or `qwen3.6-35b`. |
| `--request-delay` | Base delay used for exponential retry backoff. Default: `1` second. |
| `--max-retries` | Maximum number of attempts for transient backend errors. Default: `2`. |
| `--max-output-tokens` | Maximum API output tokens. Default: `8000`. |
| `--output-json` | JSON output path. Default: `results_with_images.json`. |
| `--output-csv` | Compact user-facing CSV output path. Default: `results.csv`. |
| `--output-debug-csv` | Technical debug CSV for partial, failed, warning, backend-error, or invalid-image cases. Default: `results_debug_failed.csv`. |
| `--strict` | Exit with code `1` for strict failure conditions. |
| `--log-level` | Logging level such as `DEBUG`, `INFO`, or `WARNING`. |

### Full-repo-test options

| Argument | Description |
|---|---|
| `--full-repo-test` | Enables the multi-repository workflow instead of single-workspace mode. |
| `--org` | GitHub organization. Default: `opentelekomcloud-docs`. |
| `--repo-limit` | Maximum repositories returned by `gh`. Default: `105`. |
| `--clone-base` | Base directory for repository clones. Default: `~/repotesting`. |
| `--result-base` | Base directory for per-repository results. Default: `~/AssetGuard/repo_results`. |
| `--env-file` | Bash environment file. Default: `.rst_checker__env`. |
| `--max-workers` | Number of parallel worker processes. Default: `8`. |
| `--worker-start-delay` | Delay between scheduling workers. Default: `0.5` seconds. |

## Output files

### JSON report

The default file `results_with_images.json` contains one entry per processed RST file with image references. Each entry contains:

- `file_path` – workspace-relative document path where possible.
- `title` – first detected RST title, if available.
- `image_count` – number of extracted image references.
- `image_refs` – extracted references, paths, line numbers, validation status, and errors.
- `status` – API status, retry information, attached image count, and errors or warnings.
- `summary` – counts of `pass`, `partial`, and `fail` results.
- `results` – enriched per-image model or local results.

Each enriched result includes `overall_score` and `verdict` in addition to the structured evaluation fields.

### Compact result CSV

The default file `results.csv` is the primary human-readable result. It contains one row per evaluated image with these columns:

- `document` – RST document path;
- `title` – detected document title;
- `image` – image filename;
- `image_type` – `screenshot`, `icon`, or `other`;
- `result` – `pass`, `partial`, or `fail`;
- `score` – normalized overall score;
- `reasons` – concise explanation of the evaluation;
- `missing_evidence` – missing or unclear evidence identified during evaluation.

If processing fails before a model result can be produced, the failure is still represented in this CSV.

### Debug CSV

The default file `results_debug_failed.csv` is intended for troubleshooting and manual investigation. It does **not** contain ordinary clean `pass` rows. It contains rows for:

- `partial` results;
- `fail` results;
- processing warnings;
- backend/API errors;
- invalid image references.

It includes the detailed criterion scores, RST reference type and line, hard-fail information, processing errors and warnings, API HTTP status, finish reason, retry attempt, reasons, and missing evidence.

### Full-repo result structure

For a successful full-repo audit, each repository receives exactly three result files:

```text
<result-base>/
├── <repository>/
│   ├── results_with_images.json
│   ├── results.csv
│   └── results_debug_failed.csv
└── <another-repository>/
    └── ...
```

Repository-level failures such as clone failures or a missing `umn/source` directory are reported in the application log. They do not create the old `run.log`, `duration_seconds.txt`, or `failed_repos.txt` artifacts.

## Strict mode behavior

When `--strict` is enabled, AssetGuard exits with status code `1` if any processed row contains one of these conditions:

- invalid image reference (`kein valides Bild`);
- `backend_error`;
- an invalid or incomplete parsed result structure;
- `hard_fail == true`;
- final verdict `fail`.

A `partial` verdict alone does not trigger strict-mode failure. However, a partial result may still be useful for manual review and is counted as a flagged file in the log output.

## Retry and API behavior

Transient HTTP statuses `429`, `500`, `502`, `503`, and `504` are retried up to the configured number of attempts. Connection and timeout errors are also retried. `--request-delay` is the base for exponential backoff: with a value of `1`, attempts wait 1 second, 2 seconds, 4 seconds, and so on. The default connection timeout is 10 seconds, and the maximum read timeout is 240 seconds.

The request uses bearer authentication, zero temperature, the OpenAI-compatible Chat Completions `messages` format, Base64 `image_url` content, and `response_format` with a strict JSON schema. The endpoint should therefore be the Chat Completions route, typically `/v1/chat/completions`. The API prompt asks for one result per attached image and requires the document and image paths to be returned exactly as provided.

After parsing, AssetGuard normalizes the model results against the list of images that were actually attached. Duplicate results are removed by `image_path`, results for unexpected image paths are discarded, and a warning is recorded if an attached image has no unique valid result.

If the API returns no valid structured result, the report records the processing state and strict mode can fail the run because the result structure is invalid.

## Path resolution rules

AssetGuard resolves paths as follows:

- Relative paths are resolved relative to the directory of the `.rst` file.
- Paths beginning with `/` are resolved relative to `--source-root` when provided.
- If no `--source-root` is provided, leading-slash paths are resolved relative to `--workspace`.
- Remote targets (`http://`, `https://`, and `data:`) are rejected as non-local images.

The resolved path is recorded in the JSON metadata. This makes it possible to diagnose references that work in one repository layout but not another.

## Automation and CI/CD integration

AssetGuard is designed so that callers can treat the CLI exit status and output files as the integration contract.

A typical CI invocation is:

```bash
python3 AssetGuard_chat_completions.py \
  --workspace "$WORKSPACE" \
  --source-root "$WORKSPACE/umn/source" \
  --output-json artifacts/results_with_images.json \
  --output-csv artifacts/results.csv \
  --output-debug-csv artifacts/results_debug_failed.csv \
  --strict
```

With `--strict`, exit status `1` is raised for an invalid image reference, `backend_error`, invalid/incomplete parsed result, local/model hard fail, or final `fail` verdict. A `partial` result alone does not make strict mode fail, but it is written to `results_debug_failed.csv` and counts as a flagged document.

For unattended organization-wide runs, shell tools such as `nohup`, systemd, a CI runner, or another scheduler can invoke `--full-repo-test`. Example:

```bash
nohup python3 AssetGuard_chat_completions.py \
  --full-repo-test \
  --org opentelekomcloud-docs \
  --repo-limit 105 \
  --clone-base ~/repotesting \
  --result-base ~/AssetGuard/repo_results \
  --env-file ~/.rst_checker__env \
  --max-workers 4 \
  --worker-start-delay 1 \
  --max-retries 2 \
  --request-delay 1 \
  --log-level INFO \
  > ~/AssetGuard/full_repo_test.log 2>&1 &
```

For automation, prefer consuming `results_with_images.json`. Use `results.csv` for human review and `results_debug_failed.csv` for focused investigation.

## Architecture and extension points

The processing flow is:

```text
RST discovery
  -> image-reference extraction and path resolution
  -> local image validation
  -> forbidden-color hard-fail check
  -> unique valid images encoded as Base64
  -> Chat Completions multimodal request
  -> structured JSON parsing
  -> result normalization/deduplication
  -> scoring and verdict calculation
  -> JSON + compact CSV + debug CSV
```

Key extension points for future maintainers are:

- `VALID_IMAGE_SUFFIXES` and `MEDIA_TYPES_BY_SUFFIX` for additional local formats;
- `FORBIDDEN_COLOR_HEX` / `FORBIDDEN_COLOR_TOLERANCE` for local visual compliance rules;
- `RESPONSE_SCHEMA` and `make_prompt()` for model evaluation behavior;
- `WEIGHTS` and verdict thresholds for scoring policy;
- `build_result_csv_rows()` for the human-facing report;
- `build_debug_csv_rows()` for investigation output;
- `normalize_api_results()` for enforcing the one-result-per-attached-image contract.

The `BACKEND_REQUIRED_TOOL` placeholder remains in the source for backend compatibility, although the current Chat Completions payload does not send it. Do not remove it without confirming the backend integration requirements.

## Troubleshooting

### `Missing AI API URL` or `Missing AI API key`

Export `AI_API_URL` and `AI_API_KEY`, pass `--api-url` and `--api-key`, or configure the environment file used by `--full-repo-test`.

### All images are reported as `kein valides Bild`

Check the file suffix, file existence, relative path, and `--source-root`. Only PNG and JPEG files are supported.

### No files are processed

Check that the workspace contains `.rst` files and that any `--path-prefix`, `--rst-file`, or `--file-list` value points to the intended files. RST files without supported image directives are skipped by design.

### Full-repo workers fail immediately

Check that `gh` is installed and authenticated, the organization is correct, Git can clone the repositories, and each repository contains `umn/source`.

### The run is slow

Latency depends on the number and size of images, the RST content, API response time, request delay, retries, and—in full-repo mode—the worker count. Increase `--max-workers` carefully because parallel requests can increase API load and rate limiting.

### A document gets a hard fail unexpectedly

Check `image_refs` and the local image itself. The current local hard-fail check searches for the exact RGB color `#3298ff`. Also check whether the API returned `hard_fail=true`.

## Typical workflow for new contributors

1. Install Python, `requests`, and `Pillow`.
2. Configure the API endpoint, key, and model without committing secrets.
3. Run AssetGuard against one small workspace or one RST file.
4. Inspect the JSON report for detailed diagnostics.
5. Inspect `results.csv` for the compact review result.
6. Inspect `results_debug_failed.csv` only when investigating partial/fail results, warnings, backend errors, or invalid images.
7. Enable `--strict` only when the expected behavior is understood and the command should gate a pipeline.
8. For organization-wide tests, verify GitHub CLI authentication and run the full-repo workflow with a controlled worker count.

## Use cases

- Documentation image validation.
- Technical documentation QA.
- Detecting missing, weak, misleading, or semantically mismatched visuals.
- Detecting configured forbidden colors in local image assets.
- CI/CD gating with `--strict`.
- Organization-wide repository audits.
- Machine-readable review reports for downstream pipelines.
- A foundation for future visual compliance checks.
