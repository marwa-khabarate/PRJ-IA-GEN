"""Création et recherche de l’index ChromaDB du corpus courant."""

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from textwrap import wrap
import unicodedata

import chromadb
from ollama import Client

from encoders import ClipEncoder, MultilingualEncoder, REVISION


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts" / "elements.jsonl"
MODELS = ROOT / "artifacts" / "models"
KINDS = ("all", "text", "figure", "table")
LABELS = {"all": "Toutes", "text": "Texte", "figure": "Figure", "table": "Tableau"}
TOKEN = re.compile(r"[a-z0-9][a-z0-9.-]+")
SPECIFIC_ELEMENT = re.compile(r"\b(figures?|fig\.?|diagrammes?|schémas?|tables?|tableaux?)\s*(?:n[°º]\s*)?(\d+)\b", re.I)
CAPTION_NUMBER = re.compile(r"^(?:Figure|Fig\.?|Table|Tableau)\s+(\d+)", re.I)
VISUAL_QUERY = re.compile(r"\b(figures?|diagrammes?|diagrams?|schémas?|illustrations?|dessins?|graphiques?|images?)\b", re.I)
ABSTRACT = re.compile(r"\b(?:dans|selon|d['’]après|in|according to)\s+(?:(?:le\s+|the\s+|l['’]))?(?:abstract|résumé)\b", re.I)
ABSTRACT_HEADING = re.compile(r"(?:^|\n)\s*(?:Abstract|Résumé)\s*\n", re.I)


def words(text):
    without_accents = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
    return {word for word in TOKEN.findall(without_accents) if len(word) >= 3 or word.isdigit()}


def element_number(record):
    numbered = CAPTION_NUMBER.match(record.get("caption", ""))
    return numbered[1] if numbered else None


def source_title(record):
    number = element_number(record)
    title = LABELS[record["type"]]
    if number:
        title += f" {number}"
    return f"{title} · page {record['page']}"


def embedding_passages(record):
    if record["type"] == "table":
        header, *rows = record["rows"]
        intro = f"{record['caption']}\n{' | '.join(header)}"
        for start in range(0, len(rows), 4):
            yield "\n".join([intro, *[" | ".join(row) for row in rows[start:start + 4]]])
        return
    if record["type"] == "figure":
        yield f"{record['text']}\nCLIP : {record['clip_label']}"
        return
    yield from wrap(record["text"], width=780, break_long_words=False, break_on_hyphens=False)


def abstract_excerpt(text):
    heading = ABSTRACT_HEADING.search(text)
    if heading is None:
        return text
    body = text[heading.end():].strip()
    return re.split(r"\n\s*1\s+Introduction\b", body, maxsplit=1, flags=re.I)[0].strip()


class CorpusIndex:
    def __init__(self, manifest=MANIFEST):
        self.manifest = Path(manifest).resolve()
        self.embedding_model = os.environ.get("MAINTIDOC_EMBED_MODEL", MultilingualEncoder.name)
        self.ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.ollama = Client(host=self.ollama_url, timeout=180)
        self.multilingual = MultilingualEncoder(MODELS / self.embedding_model) if self.embedding_model == MultilingualEncoder.name else None
        raw = self.manifest.read_bytes()
        records = (json.loads(line) for line in raw.splitlines() if line.strip())
        self.records = {record["element_id"]: record for record in records}
        if not self.records:
            raise ValueError("Le manifeste ne contient aucun élément à indexer.")
        text_revision = MultilingualEncoder.revision if self.multilingual else self.embedding_model
        config = f"index-v2\n{text_revision}\n{REVISION}".encode()
        index_key = sha256(raw + config).hexdigest()[:20]
        self.directory = self.manifest.parent / "chroma" / index_key
        self.manifest_path = self.directory / "ready.json"
        self.clip = ClipEncoder(MODELS / "clip")

    @property
    def document_name(self):
        return next(iter(self.records.values()))["source"]

    def asset(self, record):
        path = (self.manifest.parent / record["asset"]).resolve()
        if not path.is_relative_to(self.manifest.parent):
            raise ValueError("L’image référencée doit se trouver dans le dossier des résultats.")
        return path

    def embed(self, texts, *, query=False):
        if self.multilingual:
            return self.multilingual.encode(texts)
        if query and self.embedding_model.startswith("qwen3-embedding"):
            texts = [f"Instruct: Retrieve passages, figures or tables that answer the question.\nQuery: {t}" for t in texts]
        return self.ollama.embed(model=self.embedding_model, input=texts, truncate=True).embeddings

    def connect(self, *, create=False):
        client = chromadb.PersistentClient(path=str(self.directory))
        if create:
            config = {"hnsw": {"space": "cosine", "num_threads": 2}}
            self.text = client.get_or_create_collection("document-text", configuration=config, embedding_function=None)
            self.images = client.get_or_create_collection("document-clip", configuration=config, embedding_function=None)
        else:
            self.text = client.get_collection("document-text", embedding_function=None)
            self.images = client.get_collection("document-clip", embedding_function=None)

    def build(self):
        if self.multilingual:
            self.multilingual.prepare()
        figures = [r for r in self.records.values() if r["type"] == "figure"]
        labels = {}
        image_vectors = []
        if figures:
            self.clip.prepare()
            for record in figures:
                vector = self.clip.encode_image(self.asset(record))
                label = self.clip.describe(vector)
                labels[record["element_id"]] = label
                record["clip_label"] = label
                image_vectors.append(vector.tolist())

        passages = []
        for element_id, record in self.records.items():
            for number, passage in enumerate(embedding_passages(record), 1):
                passages.append((f"{element_id}-{number}", passage, {"element_id": element_id, "type": record["type"]}))

        self.connect(create=True)
        for start in range(0, len(passages), 12):
            ids, texts, metadata = map(list, zip(*passages[start:start + 12]))
            self.text.upsert(ids=ids, documents=texts, embeddings=self.embed(texts), metadatas=metadata)
        if figures:
            self.images.upsert(ids=list(labels), embeddings=image_vectors)

        ready = {"text_vectors": self.text.count(), "image_vectors": self.images.count(), "clip_labels": labels}
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory, delete=False) as file:
            json.dump(ready, file, ensure_ascii=False, indent=2)
        Path(file.name).replace(self.manifest_path)
        return ready

    def open(self):
        if not self.manifest_path.is_file():
            raise ValueError("Index absent pour ce PDF et ce modèle. Lancez d’abord python rag.py index.")
        ready = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.connect()
        if (self.text.count(), self.images.count()) != (ready["text_vectors"], ready["image_vectors"]):
            raise ValueError("Index incomplet. Relancez python rag.py index.")
        for element_id, label in ready["clip_labels"].items():
            self.records[element_id]["clip_label"] = label
        return self

    def search(self, question, modality="all", limit=3):
        if modality not in KINDS:
            raise ValueError(f"Modalité inconnue : {modality}")
        references = {
            ("table" if match[1].lower().startswith("tab") else "figure", match[2])
            for match in SPECIFIC_ELEMENT.finditer(question)
        }
        referenced_kinds = {kind for kind, _ in references}
        if modality == "all" and len(referenced_kinds) == 1:
            modality = next(iter(referenced_kinds))
        references = {(kind, number) for kind, number in references if modality in ("all", kind)}
        abstract_ids = []
        if not references and modality in ("all", "text") and ABSTRACT.search(question):
            abstract_ids = [r["element_id"] for r in self.records.values()
                            if r["type"] == "text" and ABSTRACT_HEADING.search(r["text"])]
        if abstract_ids:
            filters = {"where": {"element_id": {"$in": abstract_ids}}}
        else:
            filters = {"where": {"type": modality}} if modality != "all" else {}
        results = self.text.query(query_embeddings=self.embed([question], query=True),
                                  n_results=min(22, self.text.count()), include=["documents", "metadatas", "distances"], **filters)
        text_scores = defaultdict(lambda: -1.0)
        excerpts = {}
        for metadata, distance, excerpt in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
            key = metadata["element_id"]
            if 1.0 - distance > text_scores[key]:
                text_scores[key] = 1.0 - distance
                excerpts[key] = excerpt

        visual_scores = {}
        if (modality == "figure" or (modality == "all" and VISUAL_QUERY.search(question))) and self.images.count():
            query = self.clip.encode_text([question])[0].tolist()
            visual = self.images.query(query_embeddings=[query], n_results=min(5, self.images.count()), include=["distances"])
            visual_scores = {key: 1.0 - distance for key, distance in zip(visual["ids"][0], visual["distances"][0])}

        query_words = words(question)
        ranked = []
        for key in text_scores.keys() | visual_scores.keys():
            record = self.records[key]
            if references and (record["type"], element_number(record)) not in references:
                continue
            candidate = f"{record['text']} {record.get('clip_label', '')}"
            overlap = len(query_words & words(candidate)) / max(len(query_words), 1)
            similarity = text_scores.get(key, 0.0)
            score = similarity + 0.18 * overlap + 0.22 * visual_scores.get(key, 0.0)
            ranked.append((score, key))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            {**self.records[key], "retrieval_excerpt": abstract_excerpt(self.records[key]["text"]) if abstract_ids else excerpts.get(key)}
            for _, key in ranked[:limit]
        ]
