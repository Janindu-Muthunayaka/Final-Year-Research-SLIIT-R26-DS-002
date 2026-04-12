#!/usr/bin/env python3
"""
TreeBuilder.py
==============
Builds a **prefix trie** (tree) of valid Sinhala words for a custom OCR system.

The tree is keyed on grapheme clusters from the 302-class whitelist, so during
OCR inference other code can walk the tree node-by-node to constrain predictions
to valid Sinhala words.

Output structure (JSON):
    Each node is:  { "f": <frequency>, "c": { <cluster>: <child_node>, ... } }
      • "f" = word frequency (> 0 means a complete word ends at this node)
      • "c" = children dict mapping a whitelist grapheme cluster → child node

Pipeline:
  1. Load the 302-class whitelist from Class_List.csv.
  2. Stream the 3.8 GB corpus line-by-line (memory-safe).
  3. Filter words: decompose via greedy longest-match into whitelist clusters.
  4. Insert every valid word into the prefix trie, accumulating frequency.
  5. Serialize the trie to Sinhala_Tree.json.

Loading the tree in other code:
    >>> from TreeBuilder import SinhalaTree
    >>> tree = SinhalaTree.load("Sinhala_Tree.json")
    >>> tree.search("සිංහල")          # returns frequency or 0
    >>> tree.get_completions("සිං")    # returns {cluster: child_node, ...}
    >>> tree.prefix_exists("සිං")      # True / False

Usage:
    python TreeBuilder.py
"""

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Install it with:  pip install tqdm")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════════════
CLASS_LIST_CSV = Path(r"E:\Sliit\Research\Repositoryv2\Datasets\Class_List.csv")
CORPUS_FILE    = Path(r"E:\Sliit\Research\Repositoryv2\Datasets\Corpus\si.txt")
OUTPUT_JSON    = Path(r"E:\Sliit\Research\Repositoryv2\Final-Year-Research-SLIIT-R26-DS-002\2-Model\Tree\Sinhala_Tree.json")


# ══════════════════════════════════════════════════════════════════
#  Trie Data Structure
# ══════════════════════════════════════════════════════════════════
class TrieNode:
    """A single node in the prefix trie."""

    __slots__ = ("children", "frequency")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}   # cluster → child
        self.frequency: int = 0                      # >0 = word ends here

    # ── Serialisation ──────────────────────────────────────────
    def to_dict(self) -> dict:
        """Recursively convert the node (and subtree) to a plain dict."""
        node: dict = {}
        if self.frequency > 0:
            node["f"] = self.frequency
        if self.children:
            node["c"] = {
                cluster: child.to_dict()
                for cluster, child in self.children.items()
            }
        return node

    @classmethod
    def from_dict(cls, data: dict) -> "TrieNode":
        """Recursively reconstruct a TrieNode from a plain dict."""
        node = cls()
        node.frequency = data.get("f", 0)
        for cluster, child_data in data.get("c", {}).items():
            node.children[cluster] = cls.from_dict(child_data)
        return node


class SinhalaTree:
    """
    Prefix trie for Sinhala words, keyed on whitelist grapheme clusters.

    This is the main class that other code should import and use.

    Example usage:
        >>> tree = SinhalaTree.load("Sinhala_Tree.json")
        >>> tree.search("බලන්න")                  # → frequency (int) or 0
        >>> tree.prefix_exists("බල")               # → True / False
        >>> tree.get_completions("බල")             # → {"න": <TrieNode>, ...}
        >>> tree.walk(["බ", "ල", "න්", "න"])      # → TrieNode or None
    """

    def __init__(self, whitelist: Optional[set] = None):
        self.root = TrieNode()
        self.whitelist: set = whitelist or set()
        self._max_cluster_len: int = (
            max(len(e) for e in self.whitelist) if self.whitelist else 1
        )
        self.total_words: int = 0      # total word tokens inserted
        self.unique_words: int = 0     # unique word types in the trie

    # ── Core operations ────────────────────────────────────────
    def insert(self, clusters: List[str], freq: int = 1) -> None:
        """Insert a word (as a list of grapheme clusters) into the trie."""
        node = self.root
        for cluster in clusters:
            if cluster not in node.children:
                node.children[cluster] = TrieNode()
            node = node.children[cluster]
        if node.frequency == 0:
            self.unique_words += 1
        node.frequency += freq
        self.total_words += freq

    def walk(self, clusters: List[str]) -> Optional[TrieNode]:
        """
        Walk the trie following the given sequence of clusters.
        Returns the final TrieNode reached, or None if the path doesn't exist.
        """
        node = self.root
        for cluster in clusters:
            node = node.children.get(cluster)
            if node is None:
                return None
        return node

    def search(self, word: str) -> int:
        """
        Look up a raw word string in the trie.
        Returns the word's frequency, or 0 if not found.
        Requires the whitelist to be loaded for segmentation.
        """
        clusters = self.segment_word(word)
        if clusters is None:
            return 0
        node = self.walk(clusters)
        return node.frequency if node else 0

    def prefix_exists(self, prefix: str) -> bool:
        """
        Check whether any word in the trie starts with *prefix*.
        Requires the whitelist to be loaded for segmentation.
        """
        clusters = self.segment_word(prefix)
        if clusters is None:
            return False
        return self.walk(clusters) is not None

    def get_completions(self, prefix: str) -> Dict[str, TrieNode]:
        """
        Given a prefix string, return the children dict of the node
        reached by that prefix.  Each key is a valid next grapheme
        cluster, and the value is the subtree TrieNode.

        Returns an empty dict if the prefix doesn't exist.
        """
        clusters = self.segment_word(prefix)
        if clusters is None:
            return {}
        node = self.walk(clusters)
        return node.children if node else {}

    def walk_clusters(self, clusters: List[str]) -> Optional[TrieNode]:
        """
        Walk the trie following a pre-segmented list of clusters.
        Same as walk() — provided for API clarity.
        """
        return self.walk(clusters)

    def get_next_clusters(self, clusters: List[str]) -> Dict[str, TrieNode]:
        """
        Given a pre-segmented list of clusters already consumed, return
        the valid next grapheme clusters and their subtrees.
        Useful during beam search in OCR inference.
        """
        node = self.walk(clusters)
        return node.children if node else {}

    # ── Grapheme segmentation ──────────────────────────────────
    def segment_word(self, word: str) -> Optional[List[str]]:
        """
        Decompose *word* into a list of whitelist grapheme clusters
        using greedy longest-match (left-to-right).

        Returns the list of clusters, or None if the word contains
        any character sequence not in the whitelist.
        """
        if not self.whitelist:
            return None

        clusters: List[str] = []
        pos = 0
        length = len(word)

        while pos < length:
            matched = False
            for end in range(min(pos + self._max_cluster_len, length), pos, -1):
                candidate = word[pos:end]
                if candidate in self.whitelist:
                    clusters.append(candidate)
                    pos = end
                    matched = True
                    break
            if not matched:
                return None  # unrecognised cluster
        return clusters

    # ── Persistence ────────────────────────────────────────────
    def save(self, path: Path) -> None:
        """Serialize the entire trie to a JSON file."""
        payload = {
            "meta": {
                "total_words": self.total_words,
                "unique_words": self.unique_words,
            },
            "tree": self.root.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        file_mb = path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Saved trie → {path}  ({file_mb:.1f} MB)")

    @classmethod
    def load(cls, path, whitelist_csv: Optional[Path] = None) -> "SinhalaTree":
        """
        Load a previously saved trie from JSON.

        Parameters
        ----------
        path : str or Path
            Path to the Sinhala_Tree.json file.
        whitelist_csv : Path, optional
            If provided, also loads the whitelist so that string-based
            search / prefix_exists / get_completions work.  If not
            provided, only cluster-list-based methods (walk, walk_clusters,
            get_next_clusters) are available.

        Returns
        -------
        SinhalaTree
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        whitelist = None
        if whitelist_csv:
            whitelist = _load_whitelist(whitelist_csv)

        tree = cls(whitelist=whitelist)
        tree.root = TrieNode.from_dict(payload["tree"])
        tree.total_words = payload.get("meta", {}).get("total_words", 0)
        tree.unique_words = payload.get("meta", {}).get("unique_words", 0)
        print(f"[INFO] Loaded trie from {path}")
        print(f"       {tree.unique_words:,} unique words, {tree.total_words:,} total tokens")
        return tree


# ══════════════════════════════════════════════════════════════════
#  Whitelist loader (module-level, used by both build and load)
# ══════════════════════════════════════════════════════════════════
def _load_whitelist(csv_path: Path) -> set:
    """
    Read Class_List.csv (tab-delimited) and return a set of all unique
    rendered strings (the grapheme clusters our OCR system can recognise).
    """
    whitelist: set = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rendered = row.get("rendered", "").strip()
            if rendered:
                whitelist.add(rendered)
    print(f"[INFO] Loaded {len(whitelist)} unique classes from whitelist.")
    return whitelist


# ══════════════════════════════════════════════════════════════════
#  Corpus processing
# ══════════════════════════════════════════════════════════════════
def process_corpus(corpus_path: Path, tree: SinhalaTree) -> dict:
    """
    Stream the corpus line-by-line, filter words, insert valid ones
    into the trie.  Returns a stats dict.
    """
    total_words = 0
    valid_words = 0
    discarded_words = 0

    file_size = corpus_path.stat().st_size
    split_pattern = re.compile(r"\s+")

    with open(corpus_path, "r", encoding="utf-8") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Processing corpus",
            miniters=1,
        ) as pbar:
            for line in f:
                pbar.update(len(line.encode("utf-8")))
                words = split_pattern.split(line.strip())

                for word in words:
                    if not word:
                        continue
                    total_words += 1

                    clusters = tree.segment_word(word)
                    if clusters is not None:
                        tree.insert(clusters)
                        valid_words += 1
                    else:
                        discarded_words += 1

    stats = {
        "total_words": total_words,
        "valid_words": valid_words,
        "discarded_words": discarded_words,
        "unique_valid_words": tree.unique_words,
    }
    print(f"\n[INFO] Corpus processing complete.")
    print(f"       Total words seen  : {total_words:,}")
    print(f"       Valid words kept   : {valid_words:,}")
    print(f"       Discarded words   : {discarded_words:,}")
    print(f"       Unique valid words : {tree.unique_words:,}")
    return stats


# ══════════════════════════════════════════════════════════════════
#  Main — build the tree
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("  TreeBuilder — Sinhala Prefix Trie Builder")
    print("=" * 60)

    # Validate paths
    if not CLASS_LIST_CSV.exists():
        print(f"[ERROR] Class list not found: {CLASS_LIST_CSV}")
        sys.exit(1)
    if not CORPUS_FILE.exists():
        print(f"[ERROR] Corpus file not found: {CORPUS_FILE}")
        sys.exit(1)

    # Step 1 — Load whitelist
    whitelist = _load_whitelist(CLASS_LIST_CSV)

    # Step 2 — Create tree
    tree = SinhalaTree(whitelist=whitelist)
    print(f"[INFO] Max cluster length: {tree._max_cluster_len} codepoints")

    # Step 3 — Process corpus and build the trie
    process_corpus(CORPUS_FILE, tree)

    # Step 4 — Save the trie
    tree.save(OUTPUT_JSON)

    # Quick sanity check
    print("\n── Quick sanity check ──")
    print(f"   Root has {len(tree.root.children)} top-level branches (clusters)")
    # Show top 5 branches by subtree word count
    branch_sizes = []
    for cluster, child in tree.root.children.items():
        # Count words in this subtree (just frequency at immediate children for speed)
        branch_sizes.append((cluster, _count_subtree(child)))
    branch_sizes.sort(key=lambda x: -x[1])
    for cluster, count in branch_sizes[:5]:
        print(f"   '{cluster}' → {count:,} words in subtree")

    print("\n[DONE] Tree built successfully.")
    print(f"       Other scripts can load it with:")
    print(f"         from TreeBuilder import SinhalaTree")
    print(f"         tree = SinhalaTree.load('{OUTPUT_JSON.name}')")


def _count_subtree(node: TrieNode) -> int:
    """Count total unique words (terminal nodes) in a subtree."""
    count = 1 if node.frequency > 0 else 0
    for child in node.children.values():
        count += _count_subtree(child)
    return count


if __name__ == "__main__":
    main()
