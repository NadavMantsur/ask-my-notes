# ask-my-notes

A tiny CLI that answers questions using **only** a local folder of markdown notes, and cites which files it used.

This is a from-scratch RAG tutorial. There is no LangChain, LlamaIndex, or Haystack. Each pipeline phase is one Python file you can read.

**Learning guide (PDF):** [docs/ask-my-notes-rag-guide.pdf](docs/ask-my-notes-rag-guide.pdf) — architecture, file tree, tools, and every phase in prose (so you can learn without opening `src/` first). Markdown source: [docs/RAG-GUIDE.md](docs/RAG-GUIDE.md).

---

## How to run

You need Python 3.11+. There is **no OpenAI (or other cloud) API key**. `ingest` embeds on disk with ONNX MiniLM. `ask` needs a local [Ollama](https://ollama.com) server with `llama3.2:1b`. `pytest -v` must pass **without** Ollama.

### 1. Create a virtual environment and install dependencies

```bash
cd /path/to/ask-my-notes
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.11 -m venv` is unavailable, `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt` also works.

### 2. Ingest the notes (build the vector index)

```bash
python -m src.cli ingest
```

This loads every `.md` file in `data/`, chunks them, embeds them with a local ONNX model (`all-MiniLM-L6-v2`), and stores vectors in `./chroma_db`. The first ingest downloads that model once. Run ingest again whenever you change notes, chunk size, overlap, or the embedding model.

### 3. Install Ollama (needed only for `ask`)

Official install (needs sudo):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Without sudo, put the binary on your PATH (for example `~/.local/bin`) as in the [Ollama Linux manual install](https://github.com/ollama/ollama/blob/main/docs/linux.md). Then pull the small chat model (about 1.3 GB; the default `llama3.2` 3B model needs more disk and RAM):

```bash
export PATH="$HOME/.local/bin:$PATH"   # skip if ollama is already on PATH
ollama serve                           # leave this running
```

In another terminal:

```bash
ollama pull llama3.2:1b
```

Optional: copy `.env.example` to `.env` to override `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`).

### 4. Ask a question

Keep the `ollama serve` terminal from step 3 open and listening. In a **second** terminal, activate the venv again, then ask:

```bash
source .venv/bin/activate
python -m src.cli ask "What is the office Wi-Fi password?"
```

Other sample questions:

```bash
python -m src.cli ask "When did Project Atlas launch?"
python -m src.cli ask "What is the Friday demo Zoom link?"
```

To see which chunks were retrieved **before** the answer:

```bash
python -m src.cli ask "What is the office Wi-Fi password?" --show-chunks
```

### 5. Optional checks

Run the test suite (no Ollama, no network):

```bash
pytest -v
```

Use a different config file (the `--config` flag goes **before** the subcommand):

```bash
python -m src.cli --config config.toml ask "What is the office Wi-Fi password?" --show-chunks
```

Rebuild the learning PDF after editing `docs/RAG-GUIDE.md`:

```bash
pip install -r requirements-docs.txt
python scripts/build_guide_pdf.py
```

### Typical flow after the first setup

1. Activate the venv: `source .venv/bin/activate`
2. If notes or chunk/embedding settings changed: `python -m src.cli ingest`
3. In one terminal, make sure `ollama serve` is running and `llama3.2:1b` is pulled — leave that terminal open
4. In a second terminal (venv activated), ask: `python -m src.cli ask "your question"`

---

## RAG walkthrough (same order as the code)

`cli.py` is the conductor. `ingest` runs phases 1–4. `ask` runs phases 5–6.

### 1. Load — `src/load.py`

**What:** Read every `.md` file in `data/` and keep the filename with the text.

**Why:** Retrieval cannot cite a file you never loaded. Skipping this phase means later steps have nothing to embed.

**Config:** `[paths] data_dir`

**Command:** `python -m src.cli ingest`

### 2. Chunk — `src/chunk.py`

**What:** Split each file into overlapping character windows (~500 chars, 80-char overlap). Every chunk stores `source` (filename) and `chunk_index`.

**Why:** One vector per whole wiki page is too vague. Overlap exists so a fact sitting on a window boundary is not cut in half.

**Config:** `[chunking] size`, `[chunking] overlap` — changing these requires a new ingest.

**Command:** `python -m src.cli ingest`

### 3. Embed — `src/embed.py`

**What:** Embed every chunk with Chroma's bundled ONNX model `all-MiniLM-L6-v2` (one batched call, no cloud). At ask time, embed the question with the **same** model. The first ingest downloads that ONNX file once into `~/.cache/chroma`.

**Why:** Nearest-neighbor search compares numbers, not strings. Mixed models make distances meaningless.

**Config:** `[embedding] model` — changing this requires a new ingest.

**Command:** `python -m src.cli ingest` (chunks) and `python -m src.cli ask "..."` (the question)

### 4. Store — `src/index.py`

**What:** Reset the Chroma collection, then upsert id / text / vector / metadata into `./chroma_db`.

**Why:** Ask is a separate command; vectors must live on disk. Resetting drops stale chunks from edited or deleted files.

**Config:** `[paths] chroma_path`, `[paths] collection_name` — changing these requires a new ingest.

**Command:** `python -m src.cli ingest`

### 5. Retrieve — `src/retrieve.py`

**What:** Embed the question, ask Chroma for the top-k nearest chunks (default 4).

**Why:** The chat model should only see the notes that might answer this question. `k=1` is brittle; `k=20` adds noise.

**Config:** `[retrieval] top_k` — change this and ask again; **no ingest**.

**Command:** `python -m src.cli ask "..." --show-chunks` (prints retrieved text *before* generation)

### 6. Prompt + generate — `src/generate.py`

**What:** Build a prompt that says: answer only from these notes, say you don't know if they are missing, never invent facts, cite filenames. Send it to local Ollama `llama3.2:1b` through the OpenAI-compatible SDK (`http://localhost:11434/v1`). No OpenAI account.

**Why:** Without that system rule the model fills gaps from training data and you cannot tell which sentences came from your wiki.

**Config:** `[generation] model`, `[generation] temperature` — change these and ask again; **no ingest**.

**Command:** `python -m src.cli ask "..."`

---

## Config

Knobs live in [`config.toml`](config.toml). Optional `OLLAMA_BASE_URL` lives in `.env` (not in TOML).

| Key | Default | Re-ingest? |
|-----|---------|------------|
| `chunking.size` | 500 | yes |
| `chunking.overlap` | 80 | yes |
| `retrieval.top_k` | 4 | no |
| `embedding.model` | all-MiniLM-L6-v2 | yes |
| `generation.model` | llama3.2:1b | no |
| `generation.temperature` | 0.0 | no |
| `paths.data_dir` | data | yes |
| `paths.chroma_path` | chroma_db | yes |
| `paths.collection_name` | notes | yes |

`OLLAMA_BASE_URL` (env, default `http://localhost:11434/v1`) is the local chat server. Changing it does not require ingest. To try a larger local model later, set `generation.model` (for example `llama3.2`) and `ollama pull` that tag.

Example: set `top_k = 8` in `config.toml`, then:

```bash
python -m src.cli ask "What is the office Wi-Fi password?" --show-chunks
```

You should see more retrieved chunks. No ingest needed.

---

## How to verify RAG is actually happening

After a real ingest (local MiniLM, no cloud key):

1. `--show-chunks` for "What is the office Wi-Fi password?" prints a `wifi-and-office.md` chunk containing `orchid-42` **before** the answer.
2. The answer cites `wifi-and-office.md` and uses the password `orchid-42`.
3. `team-rituals.md` should not dominate that retrieval (it has no password).
4. "When did Atlas launch?" should surface **March 12, 2024** and/or **March 15, 2025** (Atlas vs Atlas v2).
5. "What is the CEO's pet's name?" should be "I don't know" (not a fabricated name).
6. Onboarding may mention Wi-Fi, but the password string itself comes from `wifi-and-office.md`.
7. Set `top_k = 1`, ask with `--show-chunks` — exactly one chunk. Set `top_k = 8` — more chunks. No ingest between those two.
8. `pytest -v` is green with Ollama stopped and no cloud key.
9. With Ollama stopped, `python -m src.cli ask "..."` prints the install/`ollama pull llama3.2:1b` hint and exits non-zero.

---

## Next steps after v1

- Token- or heading-aware chunking (markdown `#` sections)
- Incremental ingest (hash files, skip unchanged)
- Hybrid search (keyword + vectors) for exact strings like passwords
- A cross-encoder reranker on the top k
- A tiny eval set (questions, expected filenames, faithfulness)
- Multi-turn chat with retrieved context per turn
- PDF/HTML loaders (ingest your own PDFs as notes — not the learning guide already in `docs/`)
- Streaming tokens
- Optional CLI overrides (`--top-k`, `--temperature`) over `config.toml`
- Rebuild the same app with **LangChain** (and optionally LlamaIndex) so each `src/` module maps onto a framework class
