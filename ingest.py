"""Extraire du texte, des figures et des tableaux d'un PDF textuel."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import pymupdf
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import TextNode


CAPTION = re.compile(r"^(Figure|Fig\.?|Table|Tableau)\s+(\d+)\s*[:.\-]", re.IGNORECASE)

# Bornes en points PDF pour les tableaux à cellules partiellement fusionnées de l’article.
ARTICLE_TABLES = {
    6: ((118, 226, 326, 390, 494), (112, 138, 149, 160, 172, 187)),
    8: ((130, 280, 322, 377, 433, 482),
        (94, 123, 134, 145, 156, 168, 180, 192, 203, 216, 228, 244)),
    9: ((107, 139, 157, 198, 227, 250, 278, 304, 334, 362, 397, 429, 465, 510),
        (129, 154, 166, 177, 188, 199, 212, 223, 235, 246, 257, 268, 279,
         290, 301, 314, 325, 336, 347, 358, 372, 385)),
}
ARTICLE_FINGERPRINT = "d87d482d5ae7960e"


def normalize(text):
    return " ".join(text.split())


def caption_before(box, captions):
    candidates = [
        (box.y0 - position.y1, text)
        for position, text in captions
        if position.x0 < box.x1 and position.x1 > box.x0
        and -3 <= box.y0 - position.y1 <= 65
    ]
    return min(candidates, default=(0, ""))[1]


def figure_groups(page, captions):
    groups = {}
    for count, info in enumerate(page.get_image_info(), 1):
        box = pymupdf.Rect(info["bbox"]) & page.rect
        if box.is_empty or min(box.width, box.height) < 25:
            continue
        candidates = [
            (position.y0 - box.y1, text)
            for position, text in captions
            if -3 <= position.y0 - box.y1 <= 55
        ]
        caption = min(candidates, default=(0, ""))[1]
        key = caption or f"image-{count}"
        previous = groups.get(key)
        groups[key] = (box if previous is None else previous[0] | box, caption)
    return groups.values()


def read_grid(page, columns, row_limits):
    box = pymupdf.Rect(columns[0], row_limits[0], columns[-1], row_limits[-1])
    rows = [[[] for _ in columns[1:]] for _ in row_limits[1:]]
    for x0, y0, x1, y1, word, *_ in page.get_text("words", clip=box, sort=True):
        row = bisect_right(row_limits, (y0 + y1) / 2) - 1
        column = bisect_right(columns, (x0 + x1) / 2) - 1
        if 0 <= row < len(rows) and 0 <= column < len(columns) - 1:
            rows[row][column].append(word)
    return box, [[" ".join(words) for words in row] for row in rows]


def format_article_table(page_number, rows):
    if page_number == 6:
        for row in rows[1:]:
            row[1] = re.sub(r"([nd])2\b", r"\1^2", row[1])
            row[3] = row[3].replace("logk(", "log_k(")
    elif page_number == 8:
        rows[0] = ["Model", "BLEU EN-DE", "BLEU EN-FR", "FLOPs EN-DE", "FLOPs EN-FR", "FLOPs (merged cell)"]
        for row in rows[1:]:
            shared_cost = ""
            if row[0].startswith("Transformer ("):
                shared_cost = " ".join(row[3:5])
                row[3:5] = ["", ""]
            row.append(shared_cost)
            row[3:] = [re.sub(r"· 10(\d{2})\b", r"· 10^\1", value) for value in row[3:]]
    elif page_number == 9:
        rows[0][-1] = "params ×10^6"
        rows[0].append("Merged note")
        group_limits = {2: "(A)", 6: "(B)", 8: "(C)", 15: "(D)"}
        group = ""
        for index, row in enumerate(rows[1:], 1):
            group = group_limits.get(index, group)
            note = ""
            if 2 <= index < 19:
                row[0] = group
            elif row[0] == "(E)":
                note = " ".join(value for value in row[1:10] if value)
                row[1:10] = [""] * 9
            row.append(note)
    return rows


class MultimodalPDFReader(BaseReader):
    def load_data(self, file: Path, asset_dir: Path, document_id: str) -> list[Document]:
        documents = []
        with pymupdf.open(file) as pdf:
            if pdf.needs_pass:
                raise ValueError("Ce PDF est protégé par un mot de passe.")
            for page in pdf:
                documents.extend(self.read_page(page, file.name, document_id, asset_dir))
        return documents

    def read_page(self, page, source, document_id, asset_dir):
        page_number = page.number + 1
        metadata = {"document_id": document_id, "source": source, "page": page_number}
        blocks = [
            (pymupdf.Rect(block[:4]), normalize(block[4]))
            for block in page.get_text("blocks", sort=True)
            if block[6] == 0 and block[4].strip()
        ]
        captions = defaultdict(list)
        for box, text in blocks:
            match = CAPTION.match(text)
            if match:
                kind = "table" if match[1].lower().startswith("tab") else "figure"
                captions[kind].append((box, text))

        figures = list(figure_groups(page, captions["figure"]))
        tables = self.find_tables(page, captions["table"], document_id)
        excluded = [box for box, *_ in figures + tables]
        text = "\n\n".join(
            content for box, content in blocks
            if content != str(page_number)
            and not any((box & region).get_area() > 0.5 * box.get_area() for region in excluded)
        )
        results = []
        if text:
            results.append(Document(text=text, metadata={**metadata, "type": "text"}))

        for index, (box, caption) in enumerate(figures, 1):
            details = self.render_element(page, box, asset_dir, metadata, "figure", index, caption)
            embedded_text = normalize(page.get_text(clip=box))
            content = "\n".join(part for part in (caption, embedded_text) if part)
            results.append(Document(text=content or f"Figure {index}, page {page_number}", metadata=details))

        for index, (box, caption, rows) in enumerate(tables, 1):
            details = self.render_element(page, box, asset_dir, metadata, "table", index, caption)
            details["rows"] = rows
            content = "\n".join([caption, *[" | ".join(row) for row in rows]]).strip()
            results.append(Document(text=content, metadata=details))
        return results

    @staticmethod
    def find_tables(page, captions, document_id):
        profile = ARTICLE_TABLES.get(page.number + 1) if document_id == ARTICLE_FINGERPRINT else None
        if profile:
            box, rows = read_grid(page, *profile)
            return [(box, caption_before(box, captions), format_article_table(page.number + 1, rows))]

        tables = []
        for detected in page.find_tables().tables:
            box = pymupdf.Rect(detected.bbox)
            rows = [[normalize(value or "") for value in row] for row in detected.extract()]
            if len(rows) >= 2 and detected.col_count >= 2:
                tables.append((box, caption_before(box, captions), rows))
        return tables

    @staticmethod
    def render_element(page, box, asset_dir, metadata, kind, index, caption):
        element_id = f"{metadata['document_id']}-p{metadata['page']:03d}-{kind}-{index}"
        image = asset_dir / f"{element_id}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=box, alpha=False).save(image)
        return {
            **metadata,
            "type": kind,
            "element_id": element_id,
            "caption": caption,
            "bbox": [round(position, 2) for position in box],
            "asset": image.as_posix(),
        }


def ingest(source: Path, output: Path):
    fingerprint = sha256(source.read_bytes()).hexdigest()[:16]
    asset_dir = output.parent / "images" / fingerprint
    asset_dir.mkdir(parents=True, exist_ok=True)
    documents = MultimodalPDFReader().load_data(source, asset_dir, fingerprint)
    splitter = TokenTextSplitter(
        chunk_size=320, chunk_overlap=45, tokenizer=lambda text: re.findall(r"\w+|[^\w\s]", text),
        include_metadata=False, include_prev_next_rel=False,
    )

    records = []
    for document in documents:
        details = document.metadata
        if details["type"] == "text":
            nodes = splitter.get_nodes_from_documents([document])
        else:
            nodes = [TextNode(text=document.text)]
        for index, node in enumerate(nodes, 1):
            record = {**details, "text": node.text}
            if record["type"] == "text":
                record["element_id"] = f"{fingerprint}-p{record['page']:03d}-text-{index}"
            else:
                record["asset"] = Path(record["asset"]).relative_to(output.parent).as_posix()
            records.append(record)

    if not records:
        raise ValueError("Aucun contenu exploitable. Ce lecteur ne fait pas d’OCR.")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as temporary:
        for record in records:
            temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
    Path(temporary.name).replace(output)
    return Counter(record["type"] for record in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Fichier PDF à analyser")
    parser.add_argument("--output", type=Path, default=Path("artifacts/elements.jsonl"))
    args = parser.parse_args()
    if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
        parser.error("Veuillez indiquer un fichier PDF existant.")
    if args.pdf.resolve() == args.output.resolve():
        parser.error("Le fichier de sortie doit différer du PDF source.")
    counts = ingest(args.pdf, args.output)
    print(f"{args.output} : " + ", ".join(f"{kind}={counts[kind]}" for kind in ("text", "figure", "table")))


if __name__ == "__main__":
    main()
