"""
Few-shot puzzle dataset for encoder mode.

Key difference from PuzzleDataset:
- Each batch item includes K demo pairs + 1 query pair
- Uses num_demos array to separate demos from queries within each puzzle
- Returns demo_inputs, demo_labels, demo_mask, inputs, labels, puzzle_identifiers
"""

import json
import os
from typing import Optional

import numpy as np
import pydantic
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID


class FewShotPuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: list[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int
    max_puzzles: Optional[int] = None
    max_demos: int = 10  # Maximum demos to include per batch item


class FewShotPuzzleDataset(IterableDataset):
    """
    Dataset for encoder mode that provides demos + query per batch item.

    Each batch item includes:
    - demo_inputs: [max_K, seq_len] - demo input grids (padded)
    - demo_labels: [max_K, seq_len] - demo output grids (padded)
    - demo_mask: [max_K] - True where demo is valid
    - inputs: [seq_len] - query input
    - labels: [seq_len] - query output (target)
    - puzzle_identifiers: int - for evaluator compatibility
    """

    def __init__(self, config: FewShotPuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata (same logic as PuzzleDataset)
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0

        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples * current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers

        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets,
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, (
            f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        )
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json")) as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
            "num_demos": None,  # NEW: number of demos per puzzle
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(
                        os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"),
                        mmap_mode=mmap_mode,
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }

    def _get_demos_for_puzzle(self, puzzle_id: int, dataset: dict) -> tuple:
        """
        Extract demos for a given puzzle.

        Returns:
            demo_inputs: [K, seq_len]
            demo_labels: [K, seq_len]
            num_demos: int
        """
        puzzle_start = dataset["puzzle_indices"][puzzle_id]
        num_demos = int(dataset["num_demos"][puzzle_id])

        # Cap to max_demos
        num_demos = min(num_demos, self.config.max_demos)

        demo_inputs = np.array(dataset["inputs"][puzzle_start : puzzle_start + num_demos])
        demo_labels = np.array(dataset["labels"][puzzle_start : puzzle_start + num_demos])

        return demo_inputs, demo_labels, num_demos

    def _collate_batch(self, batch_items: list) -> dict:
        """
        Collate variable-K demos into fixed-size tensors.

        Each batch_item: {
            demo_inputs: [K_i, seq_len],
            demo_labels: [K_i, seq_len],
            num_demos: int,
            input: [seq_len],
            label: [seq_len],
            puzzle_identifier: int
        }

        Returns batch dict with:
        - demo_inputs: [B, max_K, seq_len] (padded)
        - demo_labels: [B, max_K, seq_len] (padded)
        - demo_mask: [B, max_K] (True for valid demos)
        - inputs: [B, seq_len]
        - labels: [B, seq_len]
        - puzzle_identifiers: [B]
        """
        batch_size = len(batch_items)
        seq_len = self.metadata.seq_len
        max_k = self.config.max_demos

        # Initialize padded arrays
        demo_inputs = np.full((batch_size, max_k, seq_len), self.metadata.pad_id, dtype=np.int32)
        demo_labels = np.full((batch_size, max_k, seq_len), self.metadata.pad_id, dtype=np.int32)
        demo_mask = np.zeros((batch_size, max_k), dtype=bool)
        inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
        labels = np.zeros((batch_size, seq_len), dtype=np.int32)
        puzzle_identifiers = np.zeros((batch_size,), dtype=np.int32)

        for i, item in enumerate(batch_items):
            num_demos = item["num_demos"]
            demo_inputs[i, :num_demos] = item["demo_inputs"][:num_demos]
            demo_labels[i, :num_demos] = item["demo_labels"][:num_demos]
            demo_mask[i, :num_demos] = True
            inputs[i] = item["input"]
            labels[i] = item["label"]
            puzzle_identifiers[i] = item["puzzle_identifier"]

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            labels[labels == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        return {
            "demo_inputs": torch.from_numpy(demo_inputs),
            "demo_labels": torch.from_numpy(demo_labels),
            "demo_mask": torch.from_numpy(demo_mask),
            "inputs": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),
            "puzzle_identifiers": torch.from_numpy(puzzle_identifiers),
        }

    def _iter_test(self):
        """
        Test iteration:
        - Sequential through all puzzles
        - For each puzzle, iterate through query examples (after demos)
        - Include all demos from same puzzle
        """
        for set_name, dataset in self._data.items():
            num_puzzles = len(dataset["puzzle_identifiers"])

            # Iterate through puzzles in batches
            puzzle_idx = 0
            while puzzle_idx < num_puzzles:
                batch_items = []

                # Collect items for this batch (across all ranks)
                global_batch_items = []
                while len(global_batch_items) < self.config.global_batch_size and puzzle_idx < num_puzzles:
                    # Get demos for this puzzle
                    demo_inputs, demo_labels, num_demos = self._get_demos_for_puzzle(puzzle_idx, dataset)

                    # Get query examples (those after demos)
                    puzzle_start = dataset["puzzle_indices"][puzzle_idx]
                    puzzle_end = dataset["puzzle_indices"][puzzle_idx + 1]
                    full_num_demos = int(dataset["num_demos"][puzzle_idx])

                    query_start = puzzle_start + full_num_demos
                    query_end = puzzle_end

                    # Iterate through query examples
                    for query_idx in range(query_start, query_end):
                        if len(global_batch_items) >= self.config.global_batch_size:
                            break

                        global_batch_items.append({
                            "demo_inputs": demo_inputs,
                            "demo_labels": demo_labels,
                            "num_demos": num_demos,
                            "input": np.array(dataset["inputs"][query_idx]),
                            "label": np.array(dataset["labels"][query_idx]),
                            "puzzle_identifier": dataset["puzzle_identifiers"][puzzle_idx],
                        })

                    puzzle_idx += 1

                if not global_batch_items:
                    break

                # Select items for this rank
                global_effective_batch_size = len(global_batch_items)
                rank_start = self.config.rank * self.local_batch_size
                rank_end = min((self.config.rank + 1) * self.local_batch_size, global_effective_batch_size)

                if rank_start < global_effective_batch_size:
                    batch_items = global_batch_items[rank_start:rank_end]

                    # Pad if needed
                    while len(batch_items) < self.local_batch_size:
                        # Pad with last item (will be masked in loss)
                        pad_item = {
                            "demo_inputs": np.full((1, self.metadata.seq_len), self.metadata.pad_id, dtype=np.int32),
                            "demo_labels": np.full((1, self.metadata.seq_len), self.metadata.pad_id, dtype=np.int32),
                            "num_demos": 0,
                            "input": np.full((self.metadata.seq_len,), self.metadata.pad_id, dtype=np.int32),
                            "label": np.full((self.metadata.seq_len,), IGNORE_LABEL_ID, dtype=np.int32),
                            "puzzle_identifier": self.metadata.blank_identifier_id,
                        }
                        batch_items.append(pad_item)

                    batch = self._collate_batch(batch_items)
                    yield set_name, batch, global_effective_batch_size

    def _iter_train(self):
        """
        Training iteration:
        - Shuffle groups
        - For each group, randomly sample a puzzle
        - For each puzzle, sample query examples (not demos!)
        - Attach all demos from same puzzle to each query
        """
        for set_name, dataset in self._data.items():
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            # Filter groups if max_puzzles is set
            num_groups = dataset["group_indices"].size - 1
            if self.config.max_puzzles is not None:
                valid_groups = np.where(dataset["group_indices"][:-1] < self.config.max_puzzles)[0]
                group_order = np.concatenate(
                    [rng.permutation(valid_groups) for _i in range(self.config.epochs_per_iter)]
                )
            else:
                group_order = np.concatenate(
                    [rng.permutation(num_groups) for _i in range(self.config.epochs_per_iter)]
                )

            group_idx = 0
            while group_idx < len(group_order):
                batch_items = []

                # Collect items for this batch
                while len(batch_items) < self.config.global_batch_size and group_idx < len(group_order):
                    group_id = group_order[group_idx]

                    # Get puzzle range for this group
                    puzzle_low = dataset["group_indices"][group_id]
                    puzzle_high = dataset["group_indices"][group_id + 1]
                    if self.config.max_puzzles is not None:
                        puzzle_high = min(puzzle_high, self.config.max_puzzles)

                    if puzzle_low >= puzzle_high:
                        group_idx += 1
                        continue

                    # Random puzzle from group
                    puzzle_id = rng.integers(puzzle_low, puzzle_high)
                    group_idx += 1

                    # Get demos for this puzzle
                    demo_inputs, demo_labels, num_demos = self._get_demos_for_puzzle(puzzle_id, dataset)

                    # Get query examples (after demos)
                    puzzle_start = dataset["puzzle_indices"][puzzle_id]
                    puzzle_end = dataset["puzzle_indices"][puzzle_id + 1]
                    full_num_demos = int(dataset["num_demos"][puzzle_id])

                    query_start = puzzle_start + full_num_demos
                    num_queries = puzzle_end - query_start

                    if num_queries <= 0:
                        continue

                    # Sample queries to fill batch
                    num_to_sample = min(num_queries, self.config.global_batch_size - len(batch_items))
                    query_indices = rng.choice(num_queries, num_to_sample, replace=False) + query_start

                    for query_idx in query_indices:
                        batch_items.append({
                            "demo_inputs": demo_inputs,
                            "demo_labels": demo_labels,
                            "num_demos": num_demos,
                            "input": np.array(dataset["inputs"][query_idx]),
                            "label": np.array(dataset["labels"][query_idx]),
                            "puzzle_identifier": dataset["puzzle_identifiers"][puzzle_id],
                        })

                if len(batch_items) < self.config.global_batch_size:
                    # Drop incomplete last batch
                    break

                # Select items for this rank
                rank_start = self.config.rank * self.local_batch_size
                rank_end = (self.config.rank + 1) * self.local_batch_size
                local_batch_items = batch_items[rank_start:rank_end]

                batch = self._collate_batch(local_batch_items)
                yield set_name, batch, self.config.global_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, (
            "Multithreaded data loading is not currently supported."
        )

        self._lazy_load_dataset()

        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
