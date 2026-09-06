# V2A — Video-to-Analysis

Multimodal retrieval over video, audio and documents. Upload a recording, get a transcript,
summary, sentiment and emotion breakdown — then ask questions across everything you've ingested,
by typing or by speaking.

Built as a proof-of-concept for analysing call-centre and training recordings at scale, where the
useful signal is locked inside hours of video that nobody has time to watch.

---

## The problem

Organisations accumulate large volumes of recorded conversations — support calls, training
sessions, interviews. The content is valuable and effectively unsearchable: you cannot grep a video
file, and manually reviewing a back-catalogue does not scale.

V2A turns that archive into a queryable corpus. Ingestion is one-way (media in, structured records
out), and retrieval works across the whole collection rather than one file at a time.

## Who it's for

Operations and quality teams who need to answer questions like *"how did agents handle delivery
complaints last quarter?"* without watching every call.

---

## How it works

```
video / audio ──► moviepy ──► audio track ──► OpenAI transcription ──► transcript
                                                                          │
                pdf / docx ──► PyPDF2 · python-docx ──► text ─────────────┤
                                                                          ▼
                                                       ┌──────────────────────────────┐
                                                       │ analysis pass                │
                                                       │  · summary        (OpenAI)   │
                                                       │  · sentiment      (TextBlob) │
                                                       │  · emotion + keywords        │
                                                       └──────────────┬───────────────┘
                                                                      ▼
                                             all-MiniLM-L6-v2 ──► 384-dim embedding
                                                                      │
                              ┌──────────────────────────────────────┴──────────┐
                              ▼                                                  ▼
                    FAISS IndexFlatL2                                   SQLite video_analysis
                   (similarity search)                        (transcript, summary, sentiment,
                              │                                emotion, keywords, embedding BLOB)
                              └───────────────┬──────────────────────────┘
                                              ▼
                              question ──► retrieve top matches ──► OpenAI ──► grounded answer
                                              ▲
                                   /ask-text  │  /ask-audio (spoken question,
                                              │              transcribed first)
```

The index and the source records are persisted together: embeddings are stored as a BLOB column on
the same row as the transcript they came from, so the FAISS index is rebuilt from the database on
startup rather than being a separate artefact that can drift out of sync.

## What's technically interesting

**Embeddings live with their source rows.** `save_analysis_to_db()` writes the transcript, its
derived analysis, and the embedding in one insert. `load_faiss_index()` reconstructs
`IndexFlatL2` from those rows at boot. There's no separate index file to lose, and no path where
the vector store knows about a document the database doesn't.

**Two query surfaces over one retrieval path.** `/ask-text` and `/ask-audio` differ only in that
the audio route transcribes the question first. Both then hit the same embed → search → ground →
generate pipeline, so spoken and typed questions cannot diverge in quality.

**Mixed-modality corpus.** Video, audio, PDF and DOCX all normalise to text before embedding, so a
question can retrieve across a call recording and a policy document in the same search.

**Local embeddings, hosted generation.** `all-MiniLM-L6-v2` runs in-process — embedding a growing
archive doesn't cost per-token. Only the generation step calls out to OpenAI.

**FFmpeg resolution is platform-aware.** `get_ffmpeg_executable()` resolves the binary per operating
system rather than assuming it is on `PATH`, which is what makes the same code run on a Windows
laptop and a Linux host.

---

## Stack

| Layer | Technology |
|---|---|
| Web | Flask, Jinja templates, session auth |
| Media | moviepy, FFmpeg |
| Transcription & generation | OpenAI API |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector search | FAISS (`IndexFlatL2`) |
| NLP | TextBlob (sentiment/subjectivity), wordcloud |
| Documents | PyPDF2, python-docx |
| Storage | SQLite |
| Analysis | pandas, NumPy, matplotlib |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` `POST` | `/login` · `/logout` | Session authentication |
| `GET` `POST` | `/` | Upload and analyse media or documents |
| `GET` | `/get-video-list` | List ingested media |
| `POST` | `/ask-text` | Ask a typed question across the corpus |
| `POST` | `/ask-audio` | Ask a spoken question (transcribed, then retrieved) |
| `GET` | `/analysis-history` | Previous analyses with sentiment and keywords |
| `GET` | `/download/<filename>` | Retrieve a processed file |
| `POST` | `/clear-data` | Reset the corpus and index |

All routes except `/login` are wrapped in `@login_required`.

---

## Running it

Requires Python 3.10+ and FFmpeg on the host.

```bash
git clone https://github.com/atipradmishra/V2A_chub.git
cd V2A_chub

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env             # then fill in the values below
python app.py
```

Then open `http://127.0.0.1:5000`.

### Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Transcription, summarisation and answer generation |
| `FLASK_SECRET_KEY` | Session signing — generate a random value per deployment |
| `ADMIN_USERNAME` | Login username |
| `ADMIN_PASSWORD` | Login password |

The first run downloads the `all-MiniLM-L6-v2` weights (~90 MB) and creates `video_analysis.db`.

---

## Productionising this

This is a proof-of-concept and the gaps are deliberate rather than hidden:

- **Auth** is a single hardcoded account against environment variables. Real deployment needs a
  user store, hashed credentials and sessions that survive a restart.
- **`IndexFlatL2` is exhaustive** — every query scores against every vector. Fine at POC scale;
  past roughly 10⁵ vectors this wants an IVF or HNSW index, or a managed vector store.
- **SQLite and the FAISS rebuild are single-process.** Multi-worker deployment needs Postgres
  (pgvector would collapse the two stores into one) and an index that isn't rebuilt per process.
- **Ingestion is synchronous** — a long video blocks the request. This belongs on a task queue with
  a job status endpoint.
- **No retrieval evaluation.** There is no measurement of whether the retrieved chunks actually
  ground the answer; a labelled question set and a retrieval-quality metric are the next thing I'd add.
- **Whole transcripts are embedded as single vectors**, so retrieval is document-level rather than
  passage-level. Chunking with overlap would sharpen the grounding on long recordings.

## License

MIT
