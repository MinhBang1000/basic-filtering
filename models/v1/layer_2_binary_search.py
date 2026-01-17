# detector_bisect.py
import json
import os
from bisect import bisect_left
from collections import defaultdict

class Layer2BisectDetector:
    """
    Layer 2: Rule-based Detection using binary search (bisect)
    on sorted pattern lists bucketed by pattern length.
    """
    def __init__(self, library_path="../datasets/processed_datasets/unified_attacks.json", max_scan_chars=5000):
        self.library_path = library_path
        self.max_scan_chars = max_scan_chars  # cap scanning for latency stability
        self.buckets = {}  # length -> sorted list of patterns
        self.lengths = []  # sorted unique lengths
        self.is_built = False
        self._build()

    def _build(self):
        if not os.path.exists(self.library_path):
            print(f"[!] Library not found at {self.library_path}")
            return

        with open(self.library_path, "r", encoding="utf-8") as f:
            patterns = json.load(f)

        tmp = defaultdict(set)
        for p in patterns:
            if not isinstance(p, str):
                continue
            p = p.strip().lower()
            if not p:
                continue
            tmp[len(p)].add(p)

        self.buckets = {L: sorted(list(s)) for L, s in tmp.items()}
        self.lengths = sorted(self.buckets.keys())
        self.is_built = True
        print(f"[OK] Bisect detector ready. patterns={sum(len(v) for v in self.buckets.values())} unique_lengths={len(self.lengths)}")

    @staticmethod
    def _contains_sorted(sorted_list, item: str) -> bool:
        i = bisect_left(sorted_list, item)
        return i != len(sorted_list) and sorted_list[i] == item

    def detect(self, input_history: str):
        if not self.is_built or not input_history:
            return False, []

        text = input_history.lower()
        if self.max_scan_chars and len(text) > self.max_scan_chars:
            text = text[:self.max_scan_chars]

        n = len(text)
        matches = []

        for L in self.lengths:
            if L > n:
                break
            patterns_L = self.buckets[L]
            # slide window
            for i in range(0, n - L + 1):
                sub = text[i:i+L]
                if self._contains_sorted(patterns_L, sub):
                    matches.append({
                        "pattern": sub,
                        "start_index": i,
                        "end_index": i + L - 1
                    })

        return len(matches) > 0, matches
