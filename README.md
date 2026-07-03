# AssetGuard

AssetGuard is a Python CLI tool that checks whether `assets` referenced in reStructuredText (`.rst`) files match the surrounding document context.

It extracts `asset`references, resolves local `asset`paths, sends the RST content and attached `assets` to a multimodal AI API, and writes structured evaluation results to text and JSON files containing only `failed` and `partial` verdict results.

## Quick start

1. Install dependency

```bash
pip install requests
```

2. Create a `.env` file

```env
AI_API_URL=https://your-endpoint.example/v1/responses
AI_API_KEY=your_api_key_example
AI_MODEL=your_AI_model_name_example
```

3. Load the environment variables

```bash
set -a
source .env
set +a
```

4. Run the default script

```bash
python3 assetguard.py --workspace your_workspace_path --source-root your_source_root_path
```

5. Read results:

```
cat results_with_images.txt
```
```
cat results_with_images.debug.txt
```
```
cat results_with_images.json
```

## Prerequisites

- Python 3.10 or newer recommended
- pip
- Internet access
- Valid OpenAPI v1/responses compatible AI API URL
- Valid AI API key
- Multimodal compatible AI model behind API
- A `workspace` containing a folder/subfolders with `.rst` files 
- A `source root` containing referenced `assets` from `.rst` files

## How it works

For each `.rst` file, AssetGuard:

- Extracts `asset`references
- Resolves `asset`paths
- Tries fallback file extensions if needed
- Loads supported local `asset`files
- Converts `assets` to base64 strings
- Sends the RST content and `assets` to the AI API
- Parses the structured response
- Computes a weighted score based using the AI Valuation
- Writes two human readable `.txt` and one machine-readable `.json` output file containing only `fail` and `partial` verdicts

## Supported asset types

Recognized `assets` during path resolution:

- `.png` `.jpg` `.jpeg` `.webp` `.gif`

## Scoring and Verdict

The model evaluates each `asset` with these criteria:

- `topic_match`
- `detail_match`
- `section_relevance`
- `visual_evidence`
- `contradictions`

Weights used for criteria:

- `topic_match`: 0.30
- `detail_match`: 0.20
- `section_relevance`: 0.20
- `visual_evidence`: 0.15
- `contradictions`: 0.15

Computing the overall weighted score:

$\text{score} = \frac{\sum_k w_k \cdot c_k}{\sum_k 3 \cdot w_k}$  where $\(c_k\)$ is criterion score (0–3) and $\(w_k\)$ is corresponding weight


Verdict thresholds:

- `pass`: score >= 0.80
- `partial`: score >= 0.55 and < 0.80
- `fail`: score < 0.55

## Usage

Run on the current workspace:

```bash
python3 assetguard.py --workspace . --source-root your_source_root_path
```

Run on one file:

```bash
python3 assetguard.py \
  --workspace your_wworkspace_path\
  --source-root your_source_root_path
  --rst-file your_path_to_rst_file/example.rst
```

Run on multiple files:

```bash
python3 assetguard.py \
  --workspace your_wworkspace_path \
  --source-root your_source_root_path \
  --rst-file your_path_to_rst_file/file1.rst \
  --rst-file your_path_to_rst_file/file2.rst
```

Run on files listed in a text file:

```bash
python3 assetguard.py \
  --workspace your_wworkspace_path \
  --source-root your_source_root_path \
  --file-list rst_files.txt
```

Pass API values directly instead of using `.env`:

```bash
python3 assetguard.py \
  --workspace your_wworkspace_path \
  --source-root your_source_root_path \
  --api-url "$AI_API_URL" \
  --api-key "$AI_API_KEY" \
  --model "$AI_MODEL"
```

## Output files

`results_with_images.txt`
- Readable summary output

`results_with_images.debug.txt`
- Debug output with raw model text, image resolution details, and API response info

`results_with_images.json`
- Machine-readable structured output

## Environment variables

Supported variables:

- `AI_API_URL`
- `AI_API_KEY`
- `AI_MODEL`

Important:
The script does not load `.env` automatically.
Load it like this before running the script:

```bash
set -a
source .env
set +a
```

## Notes

- Remote `asset` references are detected but not attached as local binary files
- Duplicate `asset` paths are deduplicated before submission
- Full RST content is included in the prompt
- Larger RST files may increase token usage and API cost
- The script includes debug print statements intended for development
- Workspace path and Source root path are allowed to differ
- Use exactly the `asset` folder path referenced in the `.rst` file as source root path

## Use cases

- Documentation `asset` validation
- easy to automate validation into pipeline
- Technical content QA
- Detection of misleading or weak visuals
- Machine-readable audit output for pipelines
- Base for future brand or `asset`compliance checks
