"""
Transcript Preprocessor
=======================
Cleans raw YouTube transcripts before chunking.

YouTube auto-generated transcripts are noisy:
- No punctuation / inconsistent capitalization
- Timestamp markers ([00:03:45] or <00:03:45>)
- Filler sounds ("uh", "um", "you know")
- Run-on sentences with no boundaries

This module fixes all of that before the Embedder sees the text.
"""

import re
import nltk
from typing import Optional

# Download punkt tokenizer on first use (safe to call multiple times)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


FILLER_PATTERNS = [
    r"\buh\b",
    r"\bum\b",
    r"\byou know\b",
    r"\blike i said\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bi mean\b",
    r"\bright\?\s*",
    r"\bokay so\b",
]

TIMESTAMP_PATTERNS = [
    r"\[\d{1,2}:\d{2}(?::\d{2})?\]",   # [00:03:45] or [3:45]
    r"\<\d{1,2}:\d{2}(?::\d{2})?\>",   # <00:03:45>
    r"\(\d{1,2}:\d{2}(?::\d{2})?\)",   # (00:03:45)
    r"\d{1,2}:\d{2}:\d{2}\s*",          # bare 00:03:45 (cautious — only with seconds)
]

# Minimum word count for a chunk to be considered informative
MIN_CHUNK_WORDS = 40


class TranscriptPreprocessor:
    """
    Cleans raw YouTube transcripts and splits them into
    sentence-aware chunks ready for embedding.
    """

    def __init__(self, filler_removal: bool = True):
        self.filler_removal = filler_removal

        # Compile all patterns once
        self._timestamp_re = re.compile(
            "|".join(TIMESTAMP_PATTERNS), re.IGNORECASE
        )
        self._filler_re = re.compile(
            "|".join(FILLER_PATTERNS), re.IGNORECASE
        ) if filler_removal else None

        # Collapse 3+ spaces/newlines
        self._whitespace_re = re.compile(r"[ \t]{2,}")
        self._newline_re = re.compile(r"\n{2,}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, raw_transcript: str) -> str:
        """
        Full cleaning pipeline:
        1. Remove timestamps
        2. Remove filler words
        3. Normalize whitespace
        4. Basic sentence boundary repair (add period before capital letters
           that follow a lowercase letter with no punctuation)

        Returns clean, readable text ready for chunking.
        """
        text = raw_transcript

        # Step 1 — Strip timestamps
        text = self._timestamp_re.sub(" ", text)

        # Step 2 — Remove filler words
        if self._filler_re:
            text = self._filler_re.sub(" ", text)

        # Step 3 — Repair missing sentence boundaries
        # YouTube auto-subs: "...extraction is too fast now the grind..."
        # → "...extraction is too fast. Now the grind..."
        text = self._repair_sentence_boundaries(text)

        # Step 4 — Normalize whitespace
        text = self._whitespace_re.sub(" ", text)
        text = self._newline_re.sub("\n", text)
        text = text.strip()

        return text

    def sentence_aware_chunks(
        self,
        text: str,
        max_tokens: int = 400,
        overlap_tokens: int = 80,
        words_per_token: float = 0.75,  # conservative estimate
    ) -> list[str]:
        """
        Split a cleaned transcript into sentence-aware chunks.

        Instead of hard-cutting at N characters (which splits sentences),
        we:
        1. Tokenize into sentences with nltk
        2. Accumulate sentences until we hit the token budget
        3. Slide back by `overlap_tokens` worth of sentences

        This guarantees every chunk starts and ends on a sentence boundary.
        """
        # Convert token budget to approximate word count
        max_words = int(max_tokens * words_per_token)
        overlap_words = int(overlap_tokens * words_per_token)

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If a single sentence is too long, split it at midpoint
            if sentence_words > max_words:
                words = sentence.split()
                mid = len(words) // 2
                sentence_parts = [
                    " ".join(words[:mid]),
                    " ".join(words[mid:]),
                ]
            else:
                sentence_parts = [sentence]

            for part in sentence_parts:
                part_words = len(part.split())

                if current_word_count + part_words > max_words and current_sentences:
                    # Flush current chunk
                    chunk_text = " ".join(current_sentences)
                    if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                        chunks.append(chunk_text)

                    # Overlap: keep last N words worth of sentences
                    current_sentences, current_word_count = self._overlap_window(
                        current_sentences, overlap_words
                    )

                current_sentences.append(part)
                current_word_count += part_words

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append(chunk_text)

        return chunks

    def is_informative(self, text: str, min_coffee_signals: int = 2) -> bool:
        """
        Returns True if the text contains enough coffee-related signal
        to be worth indexing. Filters out vlogs, music videos, etc.

        We check for a minimum number of domain-relevant keywords.
        """
        COFFEE_SIGNALS = {
            "coffee", "espresso", "grind", "extraction", "bean", "roast",
            "brew", "barista", "cup", "moka", "aeropress", "pour over",
            "v60", "filter", "pressure", "temperature", "bitter", "sour",
            "crema", "dose", "ratio", "water", "tamping", "channeling",
            "TDS", "EY", "yield", "bloom", "pre-infusion", "flow rate",
        }
        lower = text.lower()
        hits = sum(1 for signal in COFFEE_SIGNALS if signal in lower)
        return hits >= min_coffee_signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_sentence_boundaries(text: str) -> str:
        """
        Heuristic: insert a period + capitalize when a lowercase word
        is immediately followed by an uppercase word with only a space,
        which is a common pattern in auto-generated transcripts.

        "the grind is too coarse Now you need to" →
        "the grind is too coarse. Now you need to"
        """
        # Pattern: lowercase word + space + Uppercase word (not acronym)
        pattern = re.compile(r"([a-z,]{3,})\s+([A-Z][a-z])")
        repaired = pattern.sub(r"\1. \2", text)
        return repaired

    @staticmethod
    def _overlap_window(
        sentences: list[str], overlap_words: int
    ) -> tuple[list[str], int]:
        """
        Walk backwards from the end of `sentences` collecting sentences
        until we have `overlap_words` words. Returns the overlap window
        and its word count.
        """
        window: list[str] = []
        word_count = 0
        for sentence in reversed(sentences):
            w = len(sentence.split())
            if word_count + w > overlap_words:
                break
            window.insert(0, sentence)
            word_count += w
        return window, word_count


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    [00:00:12] so today we're going to talk about uh espresso extraction and why your
    coffee tastes bitter um so the main thing is you know the grind size and like i said
    you want to make sure the temperature is right [00:01:05] Now if you're using a
    DeLonghi super automatic machine the grind setting is digital so you can't go as
    fine as a traditional espresso machine [00:02:10] the key insight here is that
    under-extraction produces sour flavors while over-extraction gives you bitterness
    """

    preprocessor = TranscriptPreprocessor()
    cleaned = preprocessor.clean(sample)
    print("=== CLEANED ===")
    print(cleaned)
    print()

    chunks = preprocessor.sentence_aware_chunks(cleaned, max_tokens=100, overlap_tokens=20)
    print(f"=== {len(chunks)} CHUNKS ===")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}] ({len(chunk.split())} words)")
        print(chunk)

    print(f"\nIs informative: {preprocessor.is_informative(cleaned)}")
