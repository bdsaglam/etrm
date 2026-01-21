"""
Unified evaluation script for TRM, ETRM, and ETRMTRM checkpoints.

Evaluates a trained checkpoint on the full test set (or subset).
Auto-detects model type and uses appropriate dataset.

Usage:
    # Evaluate TRM checkpoint (auto-detected)
    python evaluate_trm_checkpoint.py \
        --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071

    # Evaluate ETRM checkpoint (auto-detected from config)
    python evaluate_trm_checkpoint.py \
        --checkpoint ./checkpoints/etrm-final/F1_standard/step_50000

    # Limit to N puzzle groups for quick test
    python evaluate_trm_checkpoint.py ... --max-eval-groups 32

    # Multi-GPU evaluation
    torchrun --nproc-per-node 4 evaluate_trm_checkpoint.py ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent

# Add project root to path
sys.path.insert(0, str(REPO_ROOT))

from dataset.common import PuzzleDatasetMetadata
from dataset.fewshot_puzzle_dataset import FewShotPuzzleDataset, FewShotPuzzleDatasetConfig
from evaluators.arc import ARC
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TRM/ETRM/ETRMTRM checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (step_* file)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name (default: auto-detect from checkpoint config)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["trm", "etrm", "etrmtrm", "auto"],
        default="auto",
        help="Model type: trm, etrm, etrmtrm, or auto-detect (default: auto)",
    )
    parser.add_argument(
        "--max-eval-groups",
        type=int,
        default=None,
        help="Limit evaluation to first N puzzle groups (None = full test set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: checkpoint directory)",
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        help="Additional Hydra config overrides",
    )
    return parser.parse_args()


def load_config(config_name: str, overrides: list) -> Any:
    """Load Hydra config."""
    GlobalHydra.instance().clear()
    config_dir = REPO_ROOT / "config"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


def _get_config_extras(
    cfg: Any, drop_keys: Optional[set[str]] = None
) -> Dict[str, Any]:
    """Extract extra fields from either Pydantic or OmegaConf configs."""
    if hasattr(cfg, "__pydantic_extra__") and cfg.__pydantic_extra__ is not None:
        extras = dict(cfg.__pydantic_extra__)
    else:
        extras = OmegaConf.to_container(cfg, resolve=True)
        extras = dict(extras) if isinstance(extras, dict) else {}

    for key in drop_keys or set():
        extras.pop(key, None)

    return extras


def detect_model_type(config: Any) -> str:
    """Auto-detect model type from config."""
    arch_name = config.arch.name.lower()

    if "etrmtrm" in arch_name:
        return "etrmtrm"
    elif "etrm" in arch_name or "encoder" in arch_name:
        return "etrm"
    else:
        return "trm"


def _compute_trm_eval_counts(
    dataset_paths: list[str],
    sets: list[str],
    split: str,
    max_groups: Optional[int],
) -> tuple[int, int]:
    """Compute evaluation counts for TRM-style dataset (all augmentations)."""
    total_puzzles = 0
    total_examples = 0

    for dataset_path in dataset_paths:
        split_dir = os.path.join(dataset_path, split)
        for set_name in sets:
            group_indices = np.load(
                os.path.join(split_dir, f"{set_name}__group_indices.npy")
            )
            puzzle_indices = np.load(
                os.path.join(split_dir, f"{set_name}__puzzle_indices.npy")
            )

            num_groups = group_indices.size - 1
            if max_groups is not None:
                num_groups = min(max_groups, num_groups)

            num_puzzles = int(group_indices[num_groups])
            total_puzzles += num_puzzles

            example_counts = (
                puzzle_indices[1 : num_puzzles + 1] - puzzle_indices[:num_puzzles]
            )
            total_examples += int(example_counts.sum())

    return total_puzzles, total_examples


def _compute_fewshot_eval_counts(
    dataset_paths: list[str],
    sets: list[str],
    split: str,
    max_groups: Optional[int],
) -> tuple[int, int]:
    """Compute evaluation counts for few-shot dataset (queries only)."""
    total_puzzles = 0
    total_queries = 0

    for dataset_path in dataset_paths:
        split_dir = os.path.join(dataset_path, split)
        for set_name in sets:
            group_indices = np.load(os.path.join(split_dir, f"{set_name}__group_indices.npy"))
            puzzle_indices = np.load(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"))
            num_demos = np.load(os.path.join(split_dir, f"{set_name}__num_demos.npy"))

            num_groups = group_indices.size - 1
            if max_groups is not None:
                num_groups = min(max_groups, num_groups)

            num_puzzles = int(group_indices[num_groups])
            total_puzzles += num_puzzles

            puzzle_sizes = puzzle_indices[1:num_puzzles + 1] - puzzle_indices[:num_puzzles]
            queries = puzzle_sizes - num_demos[:num_puzzles]
            total_queries += int(np.maximum(queries, 0).sum())

    return total_puzzles, total_queries


def create_dataloader(
    config: Any,
    rank: int,
    world_size: int,
    max_eval_groups: Optional[int],
    model_type: str,
):
    """Create evaluation dataloader for TRM or ETRM/ETRMTRM."""
    data_paths = (
        config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    )

    if model_type == "trm":
        # TRM uses PuzzleDataset (no encoder, all augmentations)
        dataset = PuzzleDataset(
            PuzzleDatasetConfig(
                seed=config.seed,
                dataset_paths=data_paths,
                global_batch_size=config.global_batch_size,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=rank,
                num_replicas=world_size,
                max_groups=max_eval_groups,
            ),
            split="test",
        )
    else:
        # ETRM/ETRMTRM uses FewShotPuzzleDataset (with encoder)
        dataset = FewShotPuzzleDataset(
            FewShotPuzzleDatasetConfig(
                seed=config.seed,
                dataset_paths=data_paths,
                global_batch_size=config.global_batch_size,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=rank,
                num_replicas=world_size,
                max_demos=getattr(config, "max_demos", 5),
                max_groups=max_eval_groups,
            ),
            split="test",
        )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader, dataset.metadata


def create_model(
    config: Any,
    metadata: PuzzleDatasetMetadata,
    world_size: int,
):
    """Create TRM model with loss head wrapper."""
    per_gpu_batch_size = config.global_batch_size // world_size

    model_cfg = dict(
        **_get_config_extras(config.arch, drop_keys={"name", "loss"}),
        batch_size=per_gpu_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    model = loss_head_cls(
        model_cls(model_cfg), **_get_config_extras(config.arch.loss, drop_keys={"name"})
    )

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, rank: int):
    """Load model weights from checkpoint."""
    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)

    # Handle torch.compile and DDP wrapped models
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Resize puzzle embedding if needed
    puzzle_emb_name = "model.inner.puzzle_emb.weights"
    if puzzle_emb_name in state_dict and hasattr(model, "model"):
        inner = getattr(model, "model", None)
        puzzle_emb = getattr(getattr(inner, "puzzle_emb", None), "weights", None)
        if (
            puzzle_emb is not None
            and state_dict[puzzle_emb_name].shape != puzzle_emb.shape
        ):
            expected_shape = puzzle_emb.shape
            if rank == 0:
                print(
                    f"Resetting puzzle embedding due to shape mismatch. "
                    f"Found {state_dict[puzzle_emb_name].shape}, expected {expected_shape}"
                )
            state_dict[puzzle_emb_name] = (
                torch.mean(state_dict[puzzle_emb_name], dim=0, keepdim=True)
                .expand(expected_shape)
                .contiguous()
            )

    model.load_state_dict(state_dict, strict=True)

    if rank == 0:
        print("Checkpoint loaded successfully")


def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluator: ARC,
    rank: int,
    world_size: int,
    autocast_dtype: Optional[torch.dtype] = None,
    total_expected_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run evaluation loop."""
    model.eval()
    evaluator.begin_eval()

    return_keys = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    total_samples = 0
    processed_batches = 0

    progress_bar = None
    if rank == 0 and total_expected_samples is not None:
        progress_bar = tqdm.tqdm(
            total=total_expected_samples,
            unit="samples",
            desc="Eval",
            dynamic_ncols=True,
        )

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            total_samples += global_batch_size

            if progress_bar is not None:
                progress_bar.update(global_batch_size)
            elif rank == 0 and processed_batches % 10 == 0:
                print(
                    f"Processing batch {processed_batches}: {set_name}, total samples: {total_samples}"
                )

            batch = {k: v.cuda() for k, v in batch.items()}

            with (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype
                else torch.inference_mode()
            ):
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)

                while True:
                    carry, loss, metrics, preds, all_finish = model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )

                    if all_finish:
                        break

            evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch

    if progress_bar is not None:
        progress_bar.close()

    if rank == 0:
        print(f"Total samples evaluated: {total_samples}")
        print(f"Total batches: {processed_batches}")

    results = evaluator.result(save_path=None, rank=rank, world_size=world_size)
    return results or {}


def main():
    args = parse_args()

    # Initialize distributed if needed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Load config from checkpoint directory (always use checkpoint config)
    checkpoint_config_path = Path(args.checkpoint).parent / "all_config.yaml"
    if not checkpoint_config_path.is_absolute():
        checkpoint_config_path = REPO_ROOT / checkpoint_config_path

    if checkpoint_config_path.exists():
        if rank == 0:
            print(f"Using checkpoint config: {checkpoint_config_path}")
        config = OmegaConf.load(checkpoint_config_path)

        # Apply config overrides using OmegaConf's built-in override mechanism
        if args.config_overrides:
            if rank == 0:
                print(f"Applying config overrides: {args.config_overrides}")
            override_conf = OmegaConf.from_dotlist(args.config_overrides)
            config = OmegaConf.merge(config, override_conf)
    else:
        # Fallback to Hydra config
        if args.config_name is None:
            # Auto-detect config name based on checkpoint path
            if "etrm" in str(args.checkpoint).lower():
                if "etrmtrm" in str(args.checkpoint).lower():
                    config_name = "cfg_pretrain_etrmtrm_arc_agi_1"
                else:
                    config_name = "cfg_pretrain_etrm_arc_agi_1"
            else:
                config_name = "cfg_pretrain_arc_agi_1"  # TRM default
        else:
            config_name = args.config_name

        if rank == 0:
            print(f"Warning: Checkpoint config not found at {checkpoint_config_path}")
            print(f"Falling back to Hydra config: {config_name}")
        config = load_config(config_name, args.config_overrides)

    # Auto-detect model type if needed
    if args.model_type == "auto":
        model_type = detect_model_type(config)
        if rank == 0:
            print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"{model_type.upper()} Checkpoint Evaluation")
        print(f"{'=' * 60}")
        print(f"Model type: {model_type}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Max eval groups: {args.max_eval_groups or 'all'}")
        print(f"World size: {world_size}")
        print(f"Batch size: {config.global_batch_size} (global)")
        print(f"\n>>> CRITICAL ARCHITECTURE PARAMS <<<")
        if hasattr(config.arch, "L_cycles"):
            print(f"L_cycles: {config.arch.L_cycles}")
        if hasattr(config.arch, "H_cycles"):
            print(f"H_cycles: {config.arch.H_cycles}")
        if hasattr(config.arch, "L_layers"):
            print(f"L_layers: {config.arch.L_layers}")
        print(f"{'=' * 60}\n")

    # Create eval dataloader
    eval_loader, eval_metadata = create_dataloader(
        config, rank, world_size, args.max_eval_groups, model_type
    )
    data_paths = (
        config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    )

    total_expected_samples = None
    if rank == 0:
        print("Eval metadata:")
        print(f"  Vocab size: {eval_metadata.vocab_size}")
        print(f"  Seq len: {eval_metadata.seq_len}")
        print(f"  Num puzzle identifiers: {eval_metadata.num_puzzle_identifiers}")
        if args.max_eval_groups is not None:
            displayed_groups = min(args.max_eval_groups, eval_metadata.total_groups)
        else:
            displayed_groups = eval_metadata.total_groups
        print(f"  Num groups: {displayed_groups}")
        if args.max_eval_groups is not None:
            if model_type == "trm":
                total_puzzles, total_examples = _compute_trm_eval_counts(
                    dataset_paths=data_paths,
                    sets=eval_metadata.sets,
                    split="test",
                    max_groups=args.max_eval_groups,
                )
                total_expected_samples = total_examples
                print(f"  Num puzzles: {total_puzzles}")
                print(f"  Total examples: {total_examples}")
            else:
                total_puzzles, total_queries = _compute_fewshot_eval_counts(
                    dataset_paths=data_paths,
                    sets=eval_metadata.sets,
                    split="test",
                    max_groups=args.max_eval_groups,
                )
                total_expected_samples = total_queries
                print(f"  Num puzzles: {total_puzzles}")
                print(f"  Total query examples: {total_queries}")
        print()

    # Create model
    model = create_model(config, eval_metadata, world_size)
    model = model.cuda()

    # Load checkpoint
    load_checkpoint(model, args.checkpoint, rank)

    # Create evaluator
    data_path = (
        config.data_paths_test[0]
        if len(config.data_paths_test) > 0
        else config.data_paths[0]
    )
    evaluator = ARC(
        data_path=data_path,
        eval_metadata=eval_metadata,
        submission_K=2,
        pass_Ks=(1, 2, 5, 10, 100, 1000),
        aggregated_voting=True,
    )

    if rank == 0:
        print("Starting evaluation...")

    forward_dtype = getattr(config.arch, "forward_dtype", None)
    if forward_dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    elif forward_dtype == "float16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    results = evaluate_model(
        model=model,
        eval_loader=eval_loader,
        eval_metadata=eval_metadata,
        evaluator=evaluator,
        rank=rank,
        world_size=world_size,
        autocast_dtype=autocast_dtype,
        total_expected_samples=total_expected_samples,
    )

    # Print and save results
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")
        for metric, value in sorted(results.items()):
            print(f"  {metric}: {value:.4f} ({value * 100:.2f}%)")
        print(f"{'=' * 60}")

        output_dir = args.output_dir or os.path.dirname(args.checkpoint)
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_name = os.path.basename(args.checkpoint)
        step_tag = (
            checkpoint_name if checkpoint_name.startswith("step_") else "step_unknown"
        )
        if args.max_eval_groups is None:
            results_filename = f"eval_results_full_{step_tag}.json"
        else:
            results_filename = (
                f"eval_results_groups_{args.max_eval_groups}_{step_tag}.json"
            )
        results_file = os.path.join(output_dir, results_filename)
        with open(results_file, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "config": args.config_name,
                    "max_eval_groups": args.max_eval_groups,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {results_file}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
