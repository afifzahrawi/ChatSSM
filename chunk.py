from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Chunk:
    text:               str
    source_key:         str
    source_name:        str
    category:           str
    section:            str                     = ""
    part:               str                     = ""
    embedding:          Optional[np.ndarray]    = field(default=None, compare=False, repr=False)
    relates_to_acts:    List[str]               = field(default_factory=list)  # e.g., ["Companies Act 2016", "LLP Act 2012"]
    language:           str                     = "en"

@dataclass
class SearchResult:
    chunk: Chunk
    score: float