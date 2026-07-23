# Import operating-system path utilities for building file paths
import os
# Import FAISS for vector similarity search over document embeddings
import faiss
# Import NumPy for numeric arrays expected by FAISS
import numpy as np
# Import the sentence embedding model used to encode text
from sentence_transformers import SentenceTransformer

# Build absolute path to ../docs/knowledge relative to this file (ai-service/..)
DOCS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "docs", "knowledge"))
# Minimum cosine similarity required to accept a semantic hit (embeddings are normalized)
SIM_THRESHOLD = 0.28
# Default number of semantic neighbors to fetch from FAISS
DEFAULT_K = 3

# Map YOLO / user labels to extra keywords that improve semantic matching
LABEL_ALIASES = {
    # Aliases for mobile phones
    "cell phone": ["smartphone", "mobile phone", "phone", "handset", "mobile"],
    # Aliases for laptops
    "laptop": ["computer", "notebook", "pc", "notebook computer"],
    # Aliases for cups
    "cup": ["mug", "glass", "coffee cup", "tea cup"],
    # Aliases for keyboards
    "keyboard": ["typing device", "mechanical keyboard", "keypad"],
    # Aliases for computer mice
    "mouse": ["computer mouse", "pointing device"],
    # Aliases for books
    "book": ["notebook", "textbook", "manual", "handbook"],
    # Aliases for bottles
    "bottle": ["water bottle", "flask", "drink bottle"],
    # Aliases for TV / screens
    "tv": ["television", "monitor", "display", "screen"],
    # Aliases for persons
    "person": ["human", "people", "man", "woman", "employee"],
    # Aliases for access points
    "access point": ["wireless access point", "wifi ap", "wlan ap", "ap"],
    # Aliases for routers
    "router": ["network router", "gateway router", "wifi router"],
    # Aliases for switches
    "switch": ["network switch", "ethernet switch"],
    # Aliases for firewalls
    "firewall": ["network firewall", "security appliance"],
    # Aliases for servers
    "server": ["rack server", "file server", "application server"],
    # Aliases for UPS devices
    "ups": ["uninterruptible power supply", "battery backup"],
    # Aliases for patch panels
    "patch panel": ["network patch panel", "patch bay"],
}


class RAGEngine:
    """Retrieval-Augmented Generation engine backed by local text files + FAISS."""

    def __init__(self, docs_dir: str = DOCS_DIR):
        # Load a small, fast sentence-transformer model for local embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Store text chunks used for semantic search
        self.chunks: list[str] = []
        # Store source filename for each chunk (parallel to self.chunks)
        self.sources: list[str] = []
        # Map normalized document stem -> full document text (for exact label hits)
        self.doc_by_stem: dict[str, str] = {}
        # FAISS index handle; remains None until vectors are added
        self.index = None
        # Build the in-memory knowledge index immediately
        self._build(docs_dir)

    def _build(self, docs_dir: str) -> None:
        """Read all .txt knowledge files and build exact + semantic indexes."""
        # Abort early if the knowledge directory does not exist
        if not os.path.isdir(docs_dir):
            # Log the missing directory so path mistakes are obvious
            print(f"[RAG] docs dir not found: {docs_dir}")
            # Leave the engine empty but usable
            return

        # Iterate over every entry inside the knowledge directory
        for fname in os.listdir(docs_dir):
            # Only accept plain-text knowledge files
            if not fname.lower().endswith(".txt"):
                # Skip non-text files
                continue

            # Build the absolute path to the current knowledge file
            path = os.path.join(docs_dir, fname)
            try:
                # Open the file as UTF-8 text
                with open(path, "r", encoding="utf-8") as f:
                    # Read and trim surrounding whitespace
                    text = f.read().strip()
            except Exception as e:
                # Log and skip unreadable files instead of crashing
                print(f"[RAG] skip {fname}: {e}")
                # Continue with the next file
                continue

            # Ignore empty documents
            if not text:
                # Nothing to index for this file
                continue

            # Normalized stem becomes the canonical object key, e.g. "cell phone"
            stem = os.path.splitext(fname)[0].lower().strip()
            # Keep the full document text for reliable label/filename retrieval
            self.doc_by_stem[stem] = text

            # Always index the full document as one chunk (best for short KB notes)
            self.chunks.append(text)
            # Remember which file this chunk came from
            self.sources.append(fname)

            # Also index paragraph-level chunks for longer documents
            for para in text.split("\n\n"):
                # Normalize paragraph whitespace
                para = para.strip()
                # Skip tiny fragments and exact duplicates of the full text
                if len(para) >= 10 and para != text:
                    # Store the paragraph chunk
                    self.chunks.append(para)
                    # Store the matching source filename
                    self.sources.append(fname)

        # If nothing was loaded, stop before creating an empty FAISS index
        if not self.chunks:
            # Inform the user that the knowledge base is empty
            print("[RAG] no chunks loaded.")
            # Keep index as None
            return

        # Encode all chunks into L2-normalized embeddings (cosine via inner product)
        emb = self.model.encode(
            self.chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Create a flat inner-product FAISS index with the embedding dimension
        self.index = faiss.IndexFlatIP(emb.shape[1])
        # Add all chunk vectors into the index
        self.index.add(emb)
        # Log how many chunks and documents were indexed
        print(
            f"[RAG] loaded {len(self.chunks)} chunks / "
            f"{len(self.doc_by_stem)} docs from {docs_dir}"
        )

    def _expand_query(self, query: str) -> str:
        """Enrich a short label with aliases to improve semantic recall."""
        # Normalize the incoming query text
        q = query.lower().strip()
        # Look up aliases for the exact label first
        aliases = LABEL_ALIASES.get(q, [])
        # If the query is a long sentence, also collect aliases for any known label inside it
        if not aliases:
            # Check longer labels first to prefer phrases like "cell phone"
            for label in sorted(LABEL_ALIASES.keys(), key=len, reverse=True):
                # If this label appears in the sentence, use its aliases
                if label in q:
                    # Copy aliases for query expansion
                    aliases = LABEL_ALIASES[label]
                    # Stop at the first (longest) match
                    break
        # If aliases exist, append them to the original query string
        if aliases:
            # Return expanded query text
            return f"{query} " + " ".join(aliases)
        # Otherwise return the original query unchanged
        return query

    def _canonical_stem(self, query: str) -> str | None:
        """Resolve a free-text query to a knowledge-file stem when possible."""
        # Normalize query for matching
        q = query.lower().strip()
        # Direct exact match against a document stem
        if q in self.doc_by_stem:
            # Return the exact stem
            return q

        # Prefer longer stems so "cell phone" wins over a hypothetical shorter token
        for stem in sorted(self.doc_by_stem.keys(), key=len, reverse=True):
            # If the stem appears inside the query sentence, treat it as a hit
            if stem and stem in q:
                # Return the matched document stem
                return stem

        # Try alias tables next
        for label, aliases in LABEL_ALIASES.items():
            # Build the list of candidate phrases for this label
            candidates = [label, *aliases]
            # Sort candidates by length so multi-word phrases are preferred
            for cand in sorted(candidates, key=len, reverse=True):
                # Skip empty candidates
                if not cand:
                    # Continue to next candidate
                    continue
                # Match exact candidate or candidate substring inside the query
                if q == cand or cand in q:
                    # If a document exists under the canonical label, use it
                    if label in self.doc_by_stem:
                        # Return canonical label stem
                        return label
                    # Otherwise try to find any document stem related to the candidate
                    for stem in self.doc_by_stem.keys():
                        # Loose containment check between stem and candidate
                        if cand in stem or stem in cand:
                            # Return the related document stem
                            return stem
        # No filename/label resolution possible
        return None

    def retrieve(self, query: str, k: int = DEFAULT_K) -> str | None:
        """Return knowledge text for a query, or None if nothing reliable is found."""
        # Guard against empty engine state
        if not self.doc_by_stem and (self.index is None or not self.chunks):
            # Nothing available to retrieve
            return None

        # ---------- 1) Exact / label / filename hit (highest priority) ----------
        # Resolve the query to a known document stem when possible
        stem = self._canonical_stem(query)
        # If we know the document, return its full curated text immediately
        if stem and stem in self.doc_by_stem:
            # Log the deterministic knowledge hit
            print(f"[RAG] label/file hit '{query}' -> {stem}.txt")
            # Return the exact knowledge-base description
            return self.doc_by_stem[stem]

        # ---------- 2) Semantic fallback with FAISS ----------
        # If semantic index is unavailable, stop here
        if self.index is None or not self.chunks:
            # No semantic backend ready
            return None

        # Expand short labels with aliases before embedding
        full_query = self._expand_query(query)
        # Encode the query as a normalized embedding
        q = self.model.encode(
            [full_query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search top-k similar chunks
        scores, ids = self.index.search(q, min(k, len(self.chunks)))

        # Accumulate accepted chunk texts
        results: list[str] = []
        # Track source names already added to avoid duplicates
        seen_sources: set[str] = set()

        # Walk FAISS hits in rank order
        for score, idx in zip(scores[0], ids[0]):
            # FAISS may return -1 when fewer than k valid rows exist
            if idx == -1:
                # Skip invalid row
                continue
            # Convert NumPy score to plain float
            score_f = float(score)
            # Read source filename for logging and de-duplication
            src = self.sources[idx]
            # Accept only sufficiently similar chunks
            if score_f >= SIM_THRESHOLD:
                # Skip duplicate content from the same file
                if src in seen_sources:
                    # Already have this source
                    continue
                # Remember this source
                seen_sources.add(src)
                # Keep the chunk text
                results.append(self.chunks[idx])
                # Log accepted semantic hit
                print(f"[RAG] semantic hit '{query}' -> {src} (score={score_f:.3f})")
            else:
                # Log rejected low-similarity hit for tuning
                print(f"[RAG] below threshold '{query}' -> {src} (score={score_f:.3f})")

        # If nothing passed the threshold, report no context
        if not results:
            # Caller should fall back to non-RAG behavior
            return None

        # Join multiple accepted chunks with blank lines
        return "\n\n".join(results)


# Manual smoke-test entry point
if __name__ == "__main__":
    # Construct the engine (loads docs immediately)
    rag = RAGEngine()
    # Probe a few labels that should exist in the knowledge folder
    for label in ["cell phone", "laptop", "cup", "access point", "router"]:
        # Retrieve knowledge text for the label
        ctx = rag.retrieve(label)
        # Print a visual separator for readability
        print(f"\n=== {label} ===")
        # Print either the context or an explicit miss marker
        print(ctx if ctx else "(no context found)")
