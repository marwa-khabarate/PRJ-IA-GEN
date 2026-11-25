"""Encodeurs MiniLM multilingue et CLIP quantifiés pour le CPU."""

from functools import cached_property
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer


REPO = "Xenova/clip-vit-base-patch32"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"
FILES = ("tokenizer.json", "preprocessor_config.json", "onnx/text_model_quantized.onnx",
         "onnx/vision_model_quantized.onnx")
LABELS = {
    "a diagram of a transformer neural network encoder and decoder": "architecture Transformer, encodeur et décodeur",
    "a diagram showing scaled dot product attention and multi head attention": "mécanismes d’attention et attention multi-tête",
    "a neural network flowchart with connected labeled blocks and arrows": "organigramme de blocs reliés par des flèches",
    "a bar chart with an x axis and a y axis": "graphique à barres avec axes",
    "a line graph with curves and axes": "graphique à courbes avec axes",
    "a table of numerical results with rows and columns": "tableau de résultats numériques",
    "a photograph of an object or device": "photographie d’un objet ou d’un appareil",
    "an engineering schematic with symbols and connections": "schéma technique avec symboles et connexions",
}


def normalize(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def prepare_model(directory, repo, revision, files):
    for file in files:
        if not (directory / file).is_file():
            hf_hub_download(repo, file, revision=revision, local_dir=directory)


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class MultilingualEncoder:
    name = "minilm-multilingual"
    revision = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"

    def __init__(self, directory):
        self.directory = directory

    def prepare(self):
        prepare_model(self.directory, "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q", self.revision,
                      ("model_optimized.onnx", "tokenizer.json"))

    @cached_property
    def model(self):
        return cpu_session(self.directory / "model_optimized.onnx")

    @cached_property
    def tokenizer(self):
        tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        pad_token = "<pad>" if tokenizer.token_to_id("<pad>") is not None else "[PAD]"
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token)
        return tokenizer

    def encode(self, texts):
        tokens = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.asarray([token.ids for token in tokens], dtype=np.int64),
            "attention_mask": np.asarray([token.attention_mask for token in tokens], dtype=np.int64),
            "token_type_ids": np.asarray([token.type_ids for token in tokens], dtype=np.int64),
        }
        output = self.model.run(None, {item.name: feed[item.name] for item in self.model.get_inputs()})[0]
        if output.ndim == 3:
            mask = feed["attention_mask"][..., None].astype(np.float32)
            output = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        return normalize(output).tolist()


class ClipEncoder:
    def __init__(self, directory: Path):
        self.directory = directory

    def prepare(self):
        prepare_model(self.directory, REPO, REVISION, FILES)

    @cached_property
    def text_model(self):
        return cpu_session(self.directory / FILES[2])

    @cached_property
    def image_model(self):
        return cpu_session(self.directory / FILES[3])

    @cached_property
    def tokenizer(self):
        tokenizer = Tokenizer.from_file(str(self.directory / FILES[0]))
        tokenizer.enable_truncation(max_length=77)
        tokenizer.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
        return tokenizer

    @cached_property
    def preprocess(self):
        settings = json.loads((self.directory / FILES[1]).read_text(encoding="utf-8"))
        return tuple(np.asarray(settings[key], dtype=np.float32) for key in ("image_mean", "image_std"))

    @cached_property
    def label_vectors(self):
        return self.encode_text(list(LABELS))

    def encode_text(self, texts):
        tokens = self.tokenizer.encode_batch(texts)
        feed = {"input_ids": np.asarray([token.ids for token in tokens], dtype=np.int64)}
        if any(item.name == "attention_mask" for item in self.text_model.get_inputs()):
            feed["attention_mask"] = np.asarray([token.attention_mask for token in tokens], dtype=np.int64)
        return normalize(self.text_model.run(["text_embeds"], feed)[0])

    def encode_image(self, path):
        with Image.open(path) as image:
            # Ajouter les marges avant le redimensionnement pour conserver tout le schéma.
            square = ImageOps.pad(image.convert("RGB"), (224, 224), Image.Resampling.BICUBIC, color="white")
            pixels = np.asarray(square, dtype=np.float32) / 255.0
        mean, std = self.preprocess
        pixels = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...]
        return normalize(self.image_model.run(["image_embeds"], {"pixel_values": pixels})[0])[0]

    def describe(self, embedding):
        scores = self.label_vectors @ embedding
        best = int(np.argmax(scores))
        return list(LABELS.values())[best]
