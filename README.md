# mr-local-llm

### Real-Time Vision + Local RAG + LLM Pipeline
**Offline Object-Aware Knowledge Overlay for Interactive / Mixed-Reality Style Interfaces**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-AI%20Service-009688.svg)]()
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-orange.svg)]()
[![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20Canonical-purple.svg)]()
[![LLM](https://img.shields.io/badge/LLM-Ollama%20(Qwen2.5)-green.svg)]()
[![Go](https://img.shields.io/badge/Gateway-Go-00ADD8.svg)]()
[![Mode](https://img.shields.io/badge/Mode-Fully%20Local%20%2F%20Offline-critical.svg)]()

**Repository:** [AmirhosseinGhobakhloo/mr-local-llm](https://github.com/AmirhosseinGhobakhloo/mr-local-llm)

---

## Abstract

`mr-local-llm` is an **end-to-end, fully local** system that connects **computer vision**, **retrieval-augmented generation (RAG)**, and a **local large language model** into a single real-time pipeline.

A webcam frame is analyzed by **YOLOv8**. Each detected object label is resolved against a private knowledge base using a **two-tier hybrid RAG** strategy. A local LLM (**Ollama + Qwen2.5**) then produces a **short, grounded natural-language description**, which is overlaid on the live video stream.

The system is designed as a practical foundation for **HCI / Mixed Reality research prototypes**:
objects in the physical world become entry points to **controlled, private, on-device knowledge**—without cloud APIs and without unconstrained hallucination.

---

## Why this project matters

| Challenge | How this system addresses it |
|-----------|------------------------------|
| Cloud LLM dependency | Fully offline (Ollama + local embeddings) |
| Hallucinated object descriptions | **Grounded prompting**: when RAG hits, the LLM is constrained to retrieved context |
| Noisy vision labels vs. documents | Hybrid retrieval: **canonical filename match** + **semantic FAISS fallback** |
| Real-time UX blocked by slow LLM | Multi-threaded vision loop + TTL cache |
| Unstable on-screen labels near edges | Edge-aware overlay clamping |
| Separation of concerns | FastAPI AI service + optional Go reverse-proxy gateway |

This is not a single-script demo. It is a **layered architecture** suitable for lab demos, technical interviews, and extension toward wearable / MR interfaces.

---

## Key features

- **Real-time object detection** with Ultralytics YOLOv8
- **Hybrid RAG**
  1. **Canonical retrieval** — deterministic map from label → `docs/knowledge/<label>.txt`
  2. **Semantic retrieval** — FAISS (`IndexFlatIP`) + Sentence-Transformers (`all-MiniLM-L6-v2`)
  3. **Label aliases** for robust matching (e.g. phone / cell phone / smartphone)
- **Local LLM generation** via Ollama (`qwen2.5:3b`) with low temperature for stability
- **Grounded response policy** to prioritize knowledge files over model priors
- **Concurrent vision pipeline** (capture / inference / API fetch isolation + locks)
- **Response cache with TTL** to protect FPS and avoid API spam
- **Edge-aware UI labels** so captions remain readable near frame boundaries
- **Optional Go gateway** as a clean edge entrypoint (validation + reverse proxy)
- **Private-by-design**: no external LLM vendor required at runtime

---

## System architecture
```text
┌──────────────┐     labels      ┌──────────────────────────────┐
│   Webcam     │ ──────────────► │  vision/detect_live_info.py  │
│  (OpenCV)    │                 │  YOLOv8 + UI overlay         │
└──────────────┘                 └──────────────┬───────────────┘
│ HTTP POST {message: label}
▼
┌──────────────┐   optional    ┌────────────────────────────────┐
│  gateway     │ ────────────► │  ai-service (FastAPI :8000)    │
│  (Go :8080)  │  reverse      │                                │
└──────────────┘  proxy        │  rag.py  → hybrid retrieval    │
                               │  llm_client.py → Ollama        │
                               └─────┬──────────────────────────┘
                                     │
┌────────────────────────────────────┼──────────────────────────────────┐
▼                                    ▼                                  ▼
docs/knowledge/*.txt           FAISS vectors                 Ollama Qwen2.5
(source of truth)              (semantic)                    (local LLM)

### Runtime data flow (one detection)

1. YOLO detects an object → raw label (e.g. `cell phone`)
2. Vision client sends **only the label** to `/chat` (important for deterministic RAG hits)
3. RAG resolves the label:
   - exact / canonical document stem first
   - otherwise semantic search above similarity threshold
4. If context exists → **hard-grounded prompt** (use only retrieved text)
5. LLM returns 1–2 sentences
6. Overlay is drawn on the frame; result is cached (TTL) per label

---

## Technology stack

| Layer | Technology | Role |
|-------|------------|------|
| Vision | OpenCV, Ultralytics YOLOv8 | Capture, detect, render |
| API | FastAPI, Uvicorn | AI microservice |
| Retrieval | FAISS, Sentence-Transformers | Hybrid RAG |
| Generation | Ollama + Qwen2.5:3b | Local NL explanation |
| Edge / proxy | Go (`net/http`) | Optional gateway |
| Knowledge | Plaintext domain files | Controllable ground truth |
| Concurrency | Python threads + locks | Real-time responsiveness |

---

## Repository structure

text
mr-local-llm/
├── ai-service/
│   ├── main.py              # FastAPI app, grounded prompt policy
│   ├── rag.py               # Hybrid RAG engine (canonical + FAISS)
│   └── llm_client.py        # Async Ollama client
├── vision/
│   └── detect_live_info.py  # Live camera pipeline + overlay
├── gateway/
│   └── main.go              # Reverse proxy + payload validation
├── docs/
│   └── knowledge/           # Domain knowledge (.txt per concept/object)
├── .gitignore
├── requirements.txt
└── README.md

> Model weights (e.g. `yolov8n.pt`) and virtualenvs are intentionally **not** versioned.

---

## Prerequisites

- Python **3.10+**
- [Ollama](https://ollama.com) installed and running
- Webcam
- (Optional) Go 1.21+ for the gateway
- Git

Pull the local model once:

bash
ollama pull qwen2.5:3b

---

## Setup

### 1) Clone

bash
git clone https://github.com/AmirhosseinGhobakhloo/mr-local-llm.git
cd mr-local-llm

### 2) Python environment

bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

> If dependencies are split per module, install packages needed by `ai-service` and `vision`
> (FastAPI, uvicorn, httpx, ultralytics, opencv-python, sentence-transformers, faiss-cpu, …).

### 3) Knowledge base

Add grounded facts as plain text files:

text
docs/knowledge/cell phone.txt
docs/knowledge/laptop.txt
docs/knowledge/keyboard.txt

File stem ≈ YOLO label (lowercase). This enables **canonical hits**.

### 4) Run services (order matters)

**Terminal A — Ollama**

bash
ollama serve

**Terminal B — AI service**

bash
cd ai-service
uvicorn main:app --host 0.0.0.0 --port 8000

**Terminal C — (Optional) Gateway**

bash
cd gateway
go run main.go

**Terminal D — Vision UI**

bash
cd vision
python detect_live_info.py

Press `q` in the video window to quit.

---

## API

### Health

bash
curl http://127.0.0.1:8000/health

### Chat / describe label

**Windows (cmd):**

bat
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"cell phone\"}"

**Linux / macOS / Git Bash:**

bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "cell phone"}'

Example response shape:

json
{
  "response": "Short grounded description...",
  "used_rag": true
}

- `used_rag: true` → knowledge file / semantic hit used
- `used_rag: false` → fallback behavior (no reliable local context)

---

## Design decisions (engineering rationale)

### 1) Hybrid RAG instead of pure vector search
Exact object labels should map to exact operator-authored knowledge.
Canonical stem matching makes demos **deterministic**; FAISS handles paraphrase / alias variation.

### 2) Send raw labels from vision, not long English prompts
Long instructional sentences dilute embedding match and break filename equivalence.
The API is responsible for prompt construction; the camera loop only reports **what was seen**.

### 3) Grounded prompts when context exists
If private knowledge is available, the model is constrained to that context.
This is essential for research demos and enterprise trust.

### 4) Threading + cache in the vision client
LLM latency must not freeze capture/render.
Background fetch threads and per-label TTL keep the interface interactive.

### 5) Edge-aware overlay placement
In real interaction, objects are often partial or near borders.
Labels are clamped inside the frame so the UI remains readable.

### 6) Optional Go gateway
Keeps a thin, language-agnostic edge for future auth, TLS termination, auditing, or rate limiting without coupling those concerns to Python ML code.

---

## Relevance to HCI / Mixed Reality

Although this repository currently uses a 2D webcam overlay, the interaction pattern is MR-aligned:

- **Perception**: detect entities in the user’s environment
- **Association**: bind entities to curated knowledge
- **Presentation**: render concise language in situ
- **Control**: keep the knowledge boundary explicit and local

The same service boundary (`/chat` + RAG policy) can later feed AR headsets, spatial anchors, or multi-user lab setups.

---

## Current limitations

- Detector vocabulary is bounded by the YOLO model classes unless extended
- Generation quality depends on knowledge-file quality and retrieval hit rate
- Current UI is 2D overlay (not yet world-locked MR anchors)
- Evaluation is currently demo-oriented; stronger quantitative harness is planned

---

## Security & privacy notes

- Designed for **local execution**; no cloud LLM required
- Do not commit `.env`, certificates, VPN material, or proprietary corpora if policy forbids it
- Prefer a **private** repository when knowledge files are organizational
- Model weights and virtual environments are excluded via `.gitignore`

---

## Roadmap

- [ ] Docker Compose (ai-service + Ollama)
- [ ] Stronger evaluation harness (RAG hit-rate, groundedness checks)
- [ ] Multi-object scene summaries
- [ ] Configurable prompt profiles (lab demo / industrial SOP / accessibility)
- [ ] Broader detector classes + custom training
- [ ] Integration path toward MR headset clients (spatial anchors / OpenXR)

---

## Author

**Amirhossein Ghobakhloo**  
Networks and Systems / HPC background · Offline AI systems · Enterprise hybrid RAG · HCI/MR-oriented prototypes

Project: local vision–language knowledge overlay (`mr-local-llm`)

---

## License

All rights reserved for academic / portfolio use, unless a license file is added later.


---
