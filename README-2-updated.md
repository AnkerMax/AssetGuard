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
- Sends the complete RST content and valid images to a multimodal Responses-compatible endpoint.
- Requires structured JSON output from the model.
- Computes weighted scores and final verdicts per image.
- Writes JSON, complete CSV, and fail-only CSV reports.
- Supports configurable retries, delays, logging, and strict CI mode.
- Supports parallel testing of multiple repositories through `--full-repo-test`.

## Requirements

### Single-workspace mode

- Python 3.10 or newer recommended.
- `requests`.
- `Pillow`.
- A Responses-compatible multimodal API endpoint.
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

- `AI_API_URL` – API endpoint, for example `https://your-endpoint.example/v1/responses`.
- `AI_API_KEY` – API key used for the bearer-authenticated request.
- `AI_MODEL` – model name. The default is `qwen3.6-35b` if no value is supplied.

Example environment file:

```env
AI_API_URL=https://your-endpoint.example/v1/responses
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
export AI_API_URL="https://your-endpoint.example/v1/responses"
export AI_API_KEY="your_api_key"
export AI_MODEL="your_model_name"

python3 AssetGuard-5.py --workspace .
```

The default output files are:

```text
results_with_images.json
results_with_images.csv
results_with_images.failed_only.csv
```

### Audit one file

```bash
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root \
  --rst-file path/to/file.rst
```

### Run a full organization test

```bash
python3 AssetGuard-5.py \
  --full-repo-test \
  --org opentelekomcloud-docs \
  --env-file .rst_checker__env
```

The full-repo workflow lists repositories with `gh`, clones new repositories or runs `git pull --ff-only` for existing clones, and stores each repository's output below the configured result directory.

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
11. Writes aggregated JSON and flat CSV reports.

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
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --source-root /path/to/source/root
```

Without `--rst-file`, `--file-list`, or `--path-prefix`, all `.rst` files below `--workspace` are selected recursively.

### Process selected files

```bash
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --rst-file path/to/file1.rst \
  --rst-file path/to/file2.rst
```

The `--rst-file` option can be repeated. Relative paths are interpreted relative to the workspace.

### Process a file list

```bash
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --file-list rst_files.txt
```

The file must contain one path per line. Empty lines and lines beginning with `#` are ignored.

### Restrict processing by path prefix

```bash
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --path-prefix umn/source/api_management \
  --path-prefix api-ref/source
```

A file is selected when its workspace-relative path starts with at least one supplied prefix.

### Pass API values directly

```bash
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --api-url "$AI_API_URL" \
  --api-key "$AI_API_KEY" \
  --model "$AI_MODEL"
```

### Enable strict mode

```bash
python3 AssetGuard-5.py \
  --workspace /path/to/workspace \
  --strict
```

Strict mode is intended for CI/CD. The command exits with status code `1` when a strict failure condition is found.

### Full-repo-test mode

```bash
python3 AssetGuard-5.py \
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
- failures are appended to `failed_repos.txt`;
- the total duration is written to the failure log.

The script uses the organization default `opentelekom-docs`, a repository limit of `105`, eight workers, and a worker start delay of `0.5` seconds unless overridden.

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
| `--request-delay` | Delay before each API request. Default: `1` second. |
| `--max-retries` | Maximum number of attempts for transient backend errors. Default: `2`. |
| `--max-output-tokens` | Maximum API output tokens. Default: `8000`. |
| `--output-json` | JSON output path. Default: `results_with_images.json`. |
| `--output-csv` | Complete CSV output path. Default: `results_with_images.csv`. |
| `--output-failed-csv` | Fail-only CSV output path. Default: `results_with_images.failed_only.csv`. |
| `--strict` | Exit with code `1` for strict failure conditions. |
| `--log-level` | Logging level such as `DEBUG`, `INFO`, or `WARNING`. |

### Full-repo-test options

| Argument | Description |
|---|---|
| `--full-repo-test` | Enables the multi-repository workflow instead of single-workspace mode. |
| `--org` | GitHub organization. Default: `opentelekom-docs`. |
| `--repo-limit` | Maximum repositories returned by `gh`. Default: `105`. |
| `--clone-base` | Base directory for repository clones. Default: `~/repotesting`. |
| `--script-base` | Script base directory passed to workers. Default: `~/AssetGuard`. It is currently retained for workflow compatibility. |
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

### Complete CSV report

The default file `results_with_images.csv` contains one flat row per evaluated image. It includes:

- document and image paths;
- reference type and RST line;
- detected image kind;
- interactive-button fields;
- hard-fail information;
- all criterion scores;
- overall score and final verdict;
- API status and processing errors;
- reasons and missing evidence.

### Failed-only CSV

The default file `results_with_images.failed_only.csv` contains only rows whose `final_verdict` is `fail`. This file is useful for focused manual review and CI pipelines.

### Full-repo result structure

For a full-repo test, each repository receives a directory below `--result-base` containing files such as:

```text
<result-base>/
├── <repository>/
│   ├── results_with_images.json
│   ├── results_with_images.csv
│   ├── results_with_images.failed_only.csv
│   ├── run.log
│   └── duration_seconds.txt
└── failed_repos.txt
```

The exact repository directory name is derived from the repository name. `failed_repos.txt` records clone, source-root, strict-mode, and processing failures.

## Strict mode behavior

When `--strict` is enabled, AssetGuard exits with status code `1` if any processed row contains one of these conditions:

- invalid image reference (`kein valides Bild`);
- `backend_error`;
- an invalid or incomplete parsed result structure;
- `hard_fail == true`;
- final verdict `fail`.

A `partial` verdict alone does not trigger strict-mode failure. However, a partial result may still be useful for manual review and is counted as a flagged file in the log output.

## Retry and API behavior

Transient HTTP statuses `429`, `500`, `502`, `503`, and `504` are retried up to the configured number of attempts. Connection and timeout errors are also retried. The default connection timeout is 10 seconds, and the maximum read timeout is 240 seconds.

The request uses bearer authentication, zero temperature, structured JSON output, and `tool_choice: none`. The API prompt asks for one result per attached image and requires the document and image paths to be returned exactly as provided.

If the API returns no valid structured result, the report records the processing state and strict mode can fail the run because the result structure is invalid.

## Path resolution rules

AssetGuard resolves paths as follows:

- Relative paths are resolved relative to the directory of the `.rst` file.
- Paths beginning with `/` are resolved relative to `--source-root` when provided.
- If no `--source-root` is provided, leading-slash paths are resolved relative to `--workspace`.
- Remote targets (`http://`, `https://`, and `data:`) are rejected as non-local images.

The resolved path is recorded in the JSON metadata. This makes it possible to diagnose references that work in one repository layout but not another.

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
5. Inspect the complete CSV for filtering and spreadsheet-based review.
6. Use the failed-only CSV to focus on images requiring changes.
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
