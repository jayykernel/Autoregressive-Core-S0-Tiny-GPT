"""A deterministic character-level tokenizer for small educational models."""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocabulary: list[str]):
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary must contain at least two unique characters.")
        self.vocabulary = vocabulary
        self.stoi = {char: index for index, char in enumerate(vocabulary)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot build a vocabulary from empty text.")
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f"Text contains characters outside vocabulary: {unknown!r}")
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        try:
            if any(index < 0 or index >= len(self.vocabulary) for index in ids):
                raise ValueError("Token ID is outside vocabulary.")
            return "".join(self.vocabulary[index] for index in ids)
        except (IndexError, TypeError) as exc:
            raise ValueError("Token ID is outside vocabulary.") from exc

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocabulary, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
