# detector_regex.py
import json
import os
import re

class Layer2RegexDetector:
    """
    Layer 2: Rule-based Detection using a compiled regex (alternation).
    Matches literal substrings (escaped).
    """
    def __init__(self, library_path="../datasets/processed_datasets/unified_attacks.json"):
        self.library_path = library_path
        self.regex = None
        self.patterns = []
        self.is_built = False
        self._build()

    def _build(self):
        if not os.path.exists(self.library_path):
            print(f"[!] Library not found at {self.library_path}")
            return

        with open(self.library_path, "r", encoding="utf-8") as f:
            patterns = json.load(f)

        # normalize lower
        self.patterns = [p.lower() for p in patterns if isinstance(p, str) and p.strip()]
        if not self.patterns:
            print("[!] No patterns loaded.")
            return

        # Sort by length descending to prefer longer matches (optional)
        self.patterns.sort(key=len, reverse=True)

        # Escape to make them literal
        escaped = [re.escape(p) for p in self.patterns]

        # Chunking to avoid insanely long regex (safer)
        # You can tune chunk_size (e.g., 2000-8000) based on pattern count/size.
        chunk_size = 4000
        parts = []
        for i in range(0, len(escaped), chunk_size):
            parts.append("(" + "|".join(escaped[i:i+chunk_size]) + ")")

        big_pattern = "|".join(parts)
        self.regex = re.compile(big_pattern, flags=re.IGNORECASE)
        self.is_built = True
        print(f"[OK] Regex detector ready. patterns={len(self.patterns)} chunks={len(parts)}")

    def detect(self, input_history: str):
        if not self.is_built or not input_history:
            return False, []

        text = input_history.lower()
        matches = []
        for m in self.regex.finditer(text):
            # m.group(0) is the matched literal pattern (lowercased in text)
            matches.append({
                "pattern": m.group(0),
                "start_index": m.start(),
                "end_index": m.end() - 1
            })
        return len(matches) > 0, matches
