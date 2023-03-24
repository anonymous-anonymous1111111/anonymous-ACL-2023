import json
from typing import Dict, List, Optional, Set, Tuple, Union
import random
from pathlib import Path
from pydantic.main import BaseModel


   
class RelationSentence(BaseModel):
    tokens: List[str]
    head: List[int]
    tail: List[int]
    label: str
    head_id: str = ""
    tail_id: str = ""
    label_id: str = ""
    error: str = ""
    raw: str = ""
    score: float = 0.0
    zerorc_included: bool = True

    def as_tuple(self) -> Tuple[str, str, str]:
        head = " ".join([self.tokens[i] for i in self.head])
        tail = " ".join([self.tokens[i] for i in self.tail])
        return head, self.label, tail

    def as_line(self) -> str:
        return self.json() + "\n"

    def is_valid(self) -> bool:
        for x in [self.tokens, self.head, self.tail, self.label]:
            if len(x) == 0:
                return False
        for x in [self.head, self.tail]:
            if -1 in x:
                return False
        return True

    @property
    def text(self) -> str:
        return " ".join(self.tokens)
   


class Sentence(BaseModel):
    triplets: List[RelationSentence]
    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0


class ZeroRTE_Dataset(BaseModel):
    sents: List[Sentence]

    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [Sentence(**json.loads(line)) for line in f]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")




