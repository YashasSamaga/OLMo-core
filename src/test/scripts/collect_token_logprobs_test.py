"""
Tests for the data-order invariant used by collect_token_logprobs.py.

The script expects every checkpoint to see the exact same token order.
This is achieved by pinning ``epoch=1`` in every ``reshuffle()`` call and
calling ``reset()`` after iteration.  These tests verify that contract
without requiring a GPU or real data.

The ``test_v3_val_*`` tests exercise the real ``v3_small_ppl_validation``
data mix on the Weka filesystem and are skipped when the data is not available.

The ``test_source_map_*`` and ``test_canonical_sort_*`` tests verify the
per-source storage and canonical ordering logic in collect_token_logprobs.py.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from olmo_core.data import (
    DataCollator,
    DataMix,
    NumpyFSLDataLoader,
    NumpyFSLDataset,
    NumpyPaddedFSLDataset,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.collator import PaddingDirection
from scripts.analyze.collect_token_logprobs import OUTPUT_DTYPE, build_source_map

V3_DATA_ROOT = "/weka/oe-training-default/ai2-llm"
_v3_data_available = os.path.isdir(
    os.path.join(V3_DATA_ROOT, "eval-data/perplexity/v3_small_dolma2-tokenizer")
)


def _make_dataset(tmp_path: Path, num_tokens: int, sequence_length: int) -> NumpyFSLDataset:
    mmap = np.memmap(tmp_path / "tokens.npy", dtype=np.uint16, mode="w+", shape=(num_tokens,))
    mmap[:] = np.arange(num_tokens, dtype=np.uint16)
    mmap.flush()
    return NumpyFSLDataset(
        tmp_path / "tokens.npy",
        sequence_length=sequence_length,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )


def _collect_token_ids(data_loader: NumpyFSLDataLoader) -> List[List[int]]:
    """Return a list of flattened token-id lists, one per batch."""
    batches = []
    for batch in data_loader:
        batches.append(batch["input_ids"].flatten().tolist())
    return batches


@pytest.mark.parametrize("shuffle", [True, False])
def test_reshuffle_same_epoch_gives_same_order(tmp_path: Path, shuffle: bool):
    """
    Calling ``reshuffle(epoch=1)`` + ``reset()`` repeatedly must produce
    identical iteration order — this is the invariant that
    collect_token_logprobs relies on.
    """
    num_tokens = 200
    seq_len = 4
    batch_size = 8  # in tokens

    dataset = _make_dataset(tmp_path, num_tokens, seq_len)
    data_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=batch_size,
        collator=DataCollator(pad_token_id=-1),
        shuffle=shuffle,
        work_dir=tmp_path,
    )

    runs = []
    for _ in range(3):
        data_loader.reshuffle(epoch=1, in_memory=True)
        runs.append(_collect_token_ids(data_loader))
        data_loader.reset()

    # All three runs must be identical.
    assert runs[0] == runs[1]
    assert runs[1] == runs[2]


def test_reshuffle_different_epoch_gives_different_order(tmp_path: Path):
    """
    Sanity-check: different epochs *should* produce different orders when
    shuffle=True, confirming that pinning epoch=1 is necessary.
    """
    num_tokens = 200
    seq_len = 4
    batch_size = 8

    dataset = _make_dataset(tmp_path, num_tokens, seq_len)
    data_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=batch_size,
        collator=DataCollator(pad_token_id=-1),
        shuffle=True,
        work_dir=tmp_path,
    )

    data_loader.reshuffle(epoch=1, in_memory=True)
    run1 = _collect_token_ids(data_loader)
    data_loader.reset()

    data_loader.reshuffle(epoch=2, in_memory=True)
    run2 = _collect_token_ids(data_loader)
    data_loader.reset()

    # With shuffle=True and different epochs the order should differ.
    assert run1 != run2


def test_auto_increment_epoch_changes_order(tmp_path: Path):
    """
    Demonstrates the bug that the ``epoch=1`` pin fixes: calling
    ``reshuffle()`` without an explicit epoch auto-increments and changes
    the data order.
    """
    num_tokens = 200
    seq_len = 4
    batch_size = 8

    dataset = _make_dataset(tmp_path, num_tokens, seq_len)
    data_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=batch_size,
        collator=DataCollator(pad_token_id=-1),
        shuffle=True,
        work_dir=tmp_path,
    )

    # First call: epoch auto-set to 1
    data_loader.reshuffle(in_memory=True)
    run1 = _collect_token_ids(data_loader)
    data_loader.reset()

    # Second call without explicit epoch: epoch auto-increments to 2
    data_loader.reshuffle(in_memory=True)
    run2 = _collect_token_ids(data_loader)
    data_loader.reset()

    # The auto-increment causes different ordering — this is what we avoid
    # by always passing epoch=1 in the collection script.
    assert run1 != run2


# ---------------------------------------------------------------------------
# Tests using the real v3_small_ppl_validation data on Weka
# ---------------------------------------------------------------------------

def _build_v3_data_loader(
    tmp_path: Path,
    sequence_length: int = 4096,
    global_batch_size: int = 4096 * 4,
) -> NumpyFSLDataLoader:
    """Build a data loader over v3_small_ppl_validation, mirroring the collection script."""
    tokenizer = TokenizerConfig.dolma2()
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=V3_DATA_ROOT,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        work_dir=str(tmp_path),
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()  # type: ignore[assignment]
    dataset.prepare()

    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_direction=PaddingDirection.right,
        label_ignore_index=-100,
    )
    return NumpyFSLDataLoader(
        dataset,
        global_batch_size=global_batch_size,
        collator=collator,
        work_dir=str(tmp_path),
        seed=0,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
        target_device_type="cpu",
        num_workers=0,
    )


def _collect_first_n_batches(
    data_loader: NumpyFSLDataLoader, n: int
) -> List[List[int]]:
    """Collect token IDs from the first *n* batches."""
    batches = []
    for i, batch in enumerate(data_loader):
        if i >= n:
            break
        batches.append(batch["input_ids"].flatten().tolist())
    return batches


@pytest.mark.skipif(not _v3_data_available, reason="v3_small_ppl_validation data not on disk")
def test_v3_val_same_epoch_same_order(tmp_path: Path):
    """
    Two reshuffle(epoch=1) cycles over the real v3 validation data must
    yield identical batches.
    """
    n_batches = 5
    data_loader = _build_v3_data_loader(tmp_path)

    data_loader.reshuffle(epoch=1, in_memory=True)
    run1 = _collect_first_n_batches(data_loader, n_batches)
    data_loader.reset()

    data_loader.reshuffle(epoch=1, in_memory=True)
    run2 = _collect_first_n_batches(data_loader, n_batches)
    data_loader.reset()

    assert len(run1) == n_batches
    assert run1 == run2


@pytest.mark.skipif(not _v3_data_available, reason="v3_small_ppl_validation data not on disk")
def test_v3_val_instance_indices_present(tmp_path: Path):
    """
    Verify that each batch from the v3 data loader carries ``index``
    (dataset instance indices) that can be used as document identifiers.
    """
    data_loader = _build_v3_data_loader(tmp_path)
    data_loader.reshuffle(epoch=1, in_memory=True)

    batch = next(iter(data_loader))
    data_loader.reset()

    assert "index" in batch, "Batch must contain 'index' field for document identification"
    # index shape should be (num_instances_in_batch,)
    assert batch["index"].ndim == 1
    assert batch["index"].shape[0] == batch["input_ids"].shape[0]


@pytest.mark.skipif(not _v3_data_available, reason="v3_small_ppl_validation data not on disk")
def test_v3_val_indices_stable_across_cycles(tmp_path: Path):
    """
    The (instance_index, token_ids) pairs must be identical across two
    reshuffle(epoch=1) cycles — this is what lets us join log-prob dumps
    across checkpoints.
    """
    data_loader = _build_v3_data_loader(tmp_path)
    n_batches = 3

    records = []
    for _ in range(2):
        data_loader.reshuffle(epoch=1, in_memory=True)
        cycle = []
        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break
            cycle.append(
                (batch["index"].tolist(), batch["input_ids"].flatten().tolist())
            )
        records.append(cycle)
        data_loader.reset()

    for i in range(n_batches):
        assert records[0][i][0] == records[1][i][0], f"instance indices differ at batch {i}"
        assert records[0][i][1] == records[1][i][1], f"token ids differ at batch {i}"


# ---------------------------------------------------------------------------
# Tests for build_source_map
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _v3_data_available, reason="v3_small_ppl_validation data not on disk")
def test_source_map_covers_all_instances(tmp_path: Path):
    """build_source_map must return an entry for every instance in the dataset."""
    tokenizer = TokenizerConfig.dolma2()
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=V3_DATA_ROOT,
        sequence_length=4096,
        tokenizer=tokenizer,
        work_dir=str(tmp_path),
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()  # type: ignore[assignment]
    dataset.prepare()

    source_map = build_source_map(dataset)

    assert len(source_map) == len(dataset), (
        f"source_map has {len(source_map)} entries but dataset has {len(dataset)} instances"
    )
    # Every index from 0..len(dataset)-1 should be present.
    for idx in range(len(dataset)):
        assert idx in source_map, f"instance {idx} missing from source_map"


@pytest.mark.skipif(not _v3_data_available, reason="v3_small_ppl_validation data not on disk")
def test_source_map_has_expected_sources(tmp_path: Path):
    """build_source_map must produce the known v3 validation source names."""
    tokenizer = TokenizerConfig.dolma2()
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=V3_DATA_ROOT,
        sequence_length=4096,
        tokenizer=tokenizer,
        work_dir=str(tmp_path),
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()  # type: ignore[assignment]
    dataset.prepare()

    source_map = build_source_map(dataset)
    sources = set(source_map.values())

    expected = {
        "c4_en", "dolma_books", "dolma_common-crawl", "dolma_pes2o",
        "dolma_reddit", "dolma_stack", "dolma_wiki", "ice",
        "m2d2_s2orc", "pile", "wikitext_103",
    }
    assert sources == expected, f"Got sources {sources}, expected {expected}"


@pytest.mark.skipif(not _v3_data_available, reason="v3_small_ppl_validation data not on disk")
def test_source_map_contiguous_per_source(tmp_path: Path):
    """Instances from the same source should form a contiguous index range."""
    tokenizer = TokenizerConfig.dolma2()
    dataset_cfg = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=V3_DATA_ROOT,
        sequence_length=4096,
        tokenizer=tokenizer,
        work_dir=str(tmp_path),
    )
    dataset: NumpyPaddedFSLDataset = dataset_cfg.build()  # type: ignore[assignment]
    dataset.prepare()

    source_map = build_source_map(dataset)

    # Group indices by source.
    from collections import defaultdict
    indices_by_source: Dict[str, List[int]] = defaultdict(list)
    for idx, source in source_map.items():
        indices_by_source[source].append(idx)

    for source, indices in indices_by_source.items():
        indices.sort()
        # Check contiguity: max - min + 1 == count
        assert indices[-1] - indices[0] + 1 == len(indices), (
            f"Source {source} has non-contiguous indices"
        )


# ---------------------------------------------------------------------------
# Tests for canonical sort and .npz output
# ---------------------------------------------------------------------------

def _make_fake_records(
    instance_indices: List[int],
    positions: List[int],
) -> np.ndarray:
    """Create a synthetic OUTPUT_DTYPE array for testing sort/filter logic."""
    n = len(instance_indices)
    rec = np.zeros(n, dtype=OUTPUT_DTYPE)
    rec["instance_index"] = np.array(instance_indices, dtype=np.uint32)
    rec["position_in_seq"] = np.array(positions, dtype=np.uint16)
    rec["correct_logit"] = np.random.rand(n).astype(np.float16)
    rec["token_id"] = np.arange(n, dtype=np.uint32)
    return rec


def test_canonical_sort_is_deterministic():
    """
    Sorting by (instance_index, position_in_seq) must produce the same
    order regardless of the input order.
    """
    inst = [2, 0, 1, 0, 2, 1]
    pos = [1, 0, 0, 1, 0, 1]

    rec = _make_fake_records(inst, pos)

    # Shuffle the records in two different ways and sort both.
    perm1 = np.array([5, 3, 1, 4, 0, 2])
    perm2 = np.array([2, 0, 4, 1, 5, 3])

    for perm in [perm1, perm2]:
        shuffled = rec[perm]
        sort_key = np.lexsort((shuffled["position_in_seq"], shuffled["instance_index"]))
        sorted_arr = shuffled[sort_key]

        # Should always produce instance order [0,0,1,1,2,2] with positions [0,1,0,1,0,1]
        assert list(sorted_arr["instance_index"]) == [0, 0, 1, 1, 2, 2]
        assert list(sorted_arr["position_in_seq"]) == [0, 1, 0, 1, 0, 1]


def test_npz_round_trip_per_source(tmp_path: Path):
    """
    Verify that saving per-source arrays to .npz and loading them back
    preserves the data exactly.
    """
    # Simulate two sources
    source_a = _make_fake_records([0, 0, 0], [0, 1, 2])
    source_b = _make_fake_records([1, 1], [0, 1])

    out_path = tmp_path / "test_step.npz"
    np.savez_compressed(out_path, source_a=source_a, source_b=source_b)

    loaded = np.load(out_path, allow_pickle=False)
    assert set(loaded.files) == {"source_a", "source_b"}

    np.testing.assert_array_equal(loaded["source_a"]["instance_index"], [0, 0, 0])
    np.testing.assert_array_equal(loaded["source_a"]["position_in_seq"], [0, 1, 2])
    np.testing.assert_array_equal(loaded["source_b"]["instance_index"], [1, 1])
    np.testing.assert_array_equal(loaded["source_b"]["position_in_seq"], [0, 1])

    # Logit values must survive the round-trip.
    np.testing.assert_array_equal(
        loaded["source_a"]["correct_logit"], source_a["correct_logit"]
    )


def test_npz_compressed_smaller_than_raw(tmp_path: Path):
    """Compressed .npz should be smaller than an equivalent raw .npy save."""
    # Large-ish array with lots of zeros (simulating padding-free but repetitive data).
    n = 50_000
    rec = np.zeros(n, dtype=OUTPUT_DTYPE)
    rec["instance_index"] = np.arange(n, dtype=np.uint32)

    npy_path = tmp_path / "raw.npy"
    npz_path = tmp_path / "compressed.npz"

    np.save(npy_path, rec)
    np.savez_compressed(npz_path, all=rec)

    assert npz_path.stat().st_size < npy_path.stat().st_size
