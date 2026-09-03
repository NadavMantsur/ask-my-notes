# ask-my-notes

A tiny CLI that answers questions using **only** a local folder of markdown notes, and cites which files it used.

This is a from-scratch RAG tutorial. There is no LangChain, LlamaIndex, or Haystack. Each pipeline phase is one Python file you can read.

**Learning guide (PDF):** [docs/ask-my-notes-rag-guide.pdf](docs/ask-my-notes-rag-guide.pdf) — architecture, file tree, tools, and every phase in prose (so you can learn without opening `src/` first). Markdown source: [docs/RAG-GUIDE.md](docs/RAG-GUIDE.md).

---

## How to run

You need Python 3.11+ and an OpenAI API key. `ingest` and `ask` need `OPENAI_API_KEY`. `pytest -v` must pass **without** a real API key.

### 1. Create a virtual environment and install dependencies

```bash
cd /path/to/ask-my-notes
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.11 -m venv` is unavailable, `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt` also works.

### 2. Add your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=sk-...
```

`ingest` and `ask` will fail without this. Tests do **not** need a key.

### 3. Ingest the notes (build the vector index)

```bash
python -m src.cli ingest
```

This loads every `.md` file in `data/`, chunks them, embeds them with OpenAI, and stores vectors in `./chroma_db`. Run this again whenever you change notes, chunk size, overlap, or the embedding model.

### 4. Ask a question

```bash
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

Run the test suite (no API key needed):

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
3. Ask: `python -m src.cli ask "your question"`

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

**What:** Call OpenAI `text-embedding-3-small` once per ingest (batched) to turn each chunk into a vector. At ask time, embed the question with the **same** model.

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

**What:** Build a prompt that says: answer only from these notes, say you don't know if they are missing, never invent facts, cite filenames. Send it to `gpt-4o-mini`.

**Why:** Without that system rule the model fills gaps from training data and you cannot tell which sentences came from your wiki.

**Config:** `[generation] model`, `[generation] temperature` — change these and ask again; **no ingest**.

**Command:** `python -m src.cli ask "..."`

---

## Config

Knobs live in [`config.toml`](config.toml). Secrets live in `.env` (`OPENAI_API_KEY` only).

| Key | Default | Re-ingest? |
|-----|---------|------------|
| `chunking.size` | 500 | yes |
| `chunking.overlap` | 80 | yes |
| `retrieval.top_k` | 4 | no |
| `embedding.model` | text-embedding-3-small | yes |
| `generation.model` | gpt-4o-mini | no |
| `generation.temperature` | 0.0 | no |
| `paths.data_dir` | data | yes |
| `paths.chroma_path` | chroma_db | yes |
| `paths.collection_name` | notes | yes |

Example: set `top_k = 8` in `config.toml`, then:

```bash
python -m src.cli ask "What is the office Wi-Fi password?" --show-chunks
```

You should see more retrieved chunks. No ingest needed.

---

## How to verify RAG is actually happening

After a real ingest (`OPENAI_API_KEY` set):

1. `--show-chunks` for "What is the office Wi-Fi password?" prints a `wifi-and-office.md` chunk containing `orchid-42` **before** the answer.
2. The answer cites `wifi-and-office.md` and uses the password `orchid-42`.
3. `team-rituals.md` should not dominate that retrieval (it has no password).
4. "When did Atlas launch?" should surface **March 12, 2024** and/or **March 15, 2025** (Atlas vs Atlas v2).
5. "What is the CEO's pet's name?" should be "I don't know" (not a fabricated name).
6. Onboarding may mention Wi-Fi, but the password string itself comes from `wifi-and-office.md`.
7. Set `top_k = 1`, ask with `--show-chunks` — exactly one chunk. Set `top_k = 8` — more chunks. No ingest between those two.
8. `pytest -v` is green with no API key.
9. Empty `OPENAI_API_KEY` then `python -m src.cli ingest` prints the copy-`.env.example` message and exits non-zero.

---

## Next steps after v1

- Token- or heading-aware chunking (markdown `#` sections)
- Incremental ingest (hash files, skip unchanged)
- Hybrid search (keyword + vectors) for exact strings like passwords
- A cross-encoder reranker on the top k
- A tiny eval set (questions, expected filenames, faithfulness)
- Multi-turn chat with retrieved context per turn
- PDF/HTML loaders (ingest your own PDFs as notes — not the learning guide already in `docs/`)
- Local embeddings (Ollama) as an `--offline` path
- Streaming tokens
- Optional CLI overrides (`--top-k`, `--temperature`) over `config.toml`
- Rebuild the same app with **LangChain** (and optionally LlamaIndex) so each `src/` module maps onto a framework class
