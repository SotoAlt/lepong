"""Occlusion A/B matrix evaluation for the Pong state head.

Runs a matrix of evaluations: checkpoint x occlusion level.

At each cell, generates in-distribution frames (AI-tracked policy), applies
the specified occlusion to the right side of each frame, runs
encoder -> predictor -> state head, measures median absolute error on
ball_x, ball_y, pad_l, pad_r.

Key questions this answers:
  1. Does the occlusion-augmented checkpoint still work on unoccluded frames?
  2. Does the baseline checkpoint collapse under occlusion?
  3. Does data augmentation fix the pixel-level OOD failure?

Usage:
    python scripts/eval_occlusion.py \
        --checkpoints checkpoints/lepong_statehead_frozen.pt \
                      checkpoints/lepong_statehead_occ_aug.pt \
        --output results/eval_occlusion.json
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE
from model.pong_world import PongWorld


COURT_H = 0.6
BALL_SPEED_MAX = 0.025
STATE_NAMES = [
    "ball_x", "ball_y", "ball_vx", "ball_vy",
    "pad_l", "pad_r", "score_l", "score_r", "speed", "rally",
]
REPORT_DIMS = [0, 1, 4, 5]  # ball_x, ball_y, pad_l, pad_r


def generate_frames(n_episodes: int, steps: int, frameskip: int, seed: int):
    env = PongWorld()
    rng = np.random.default_rng(seed)
    frames = []
    states = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        noise = rng.uniform(0.0, 0.15)
        for step in range(steps):
            action = env.ai_action(noise=noise)
            for _ in range(frameskip):
                env.step(action)
            frames.append(env.render(128))
            states.append(env.get_state())
    return np.stack(frames, axis=0), np.stack(states, axis=0).astype(np.float32)


def apply_occlusion(frames: np.ndarray, occ: float) -> np.ndarray:
    """Return a copy of frames with the right-occ fraction blacked out."""
    if occ <= 0:
        return frames.copy()
    out = frames.copy()
    size = out.shape[2]
    start = int(size * (1 - occ))
    out[:, :, start:, :] = 0
    return out


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", 10)
    model = JEPAPool(embed_dim=ckpt.get("embed_dim", EMBED_DIM), state_dim=state_dim)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = model.to(device)
    state_mean = ckpt.get("state_mean", torch.zeros(state_dim)).to(device)
    state_std = ckpt.get("state_std", torch.ones(state_dim)).to(device)
    return model, state_mean, state_std


def eval_state_head(
    frames: np.ndarray,
    states: np.ndarray,
    model: JEPAPool,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    device: torch.device,
) -> dict:
    """For each frame-window, run state head and record per-dim abs error."""
    N = len(frames)
    i_start = HISTORY_SIZE
    i_end = N - 1
    errs = {name: [] for name in STATE_NAMES}
    BATCH = 32
    with torch.no_grad():
        for batch_start in range(i_start, i_end, BATCH):
            idxs = list(range(batch_start, min(batch_start + BATCH, i_end)))
            ctx_list = []
            tgt_list = []
            for i in idxs:
                ctx_list.append(frames[i - HISTORY_SIZE + 1 : i + 1])
                tgt_list.append(states[i + 1])
            ctx_np = np.stack(ctx_list, axis=0)
            ctx = (
                torch.from_numpy(ctx_np)
                .float()
                .permute(0, 1, 4, 2, 3)
                / 255.0
            ).to(device)
            emb = model.encode(ctx)
            B = emb.shape[0]
            zero_action = torch.zeros(B, HISTORY_SIZE, 2, device=device)
            action_emb = model.action_encoder(
                zero_action.reshape(B * HISTORY_SIZE, 2)
            ).reshape(B, HISTORY_SIZE, -1)
            pred = model.predict_next(emb, action_emb)
            s_norm = model.state_head(pred)
            s = s_norm * state_std + state_mean
            tgt = torch.from_numpy(np.stack(tgt_list)).to(device)
            abs_err = torch.abs(s - tgt).cpu().numpy()
            for dim_idx, name in enumerate(STATE_NAMES):
                errs[name].extend(abs_err[:, dim_idx].tolist())
    return {name: np.array(errs[name]) for name in STATE_NAMES}


def summarize(errs: dict) -> dict:
    out = {}
    for i in REPORT_DIMS:
        name = STATE_NAMES[i]
        arr = errs[name]
        out[name] = {
            "n": int(len(arr)),
            "median": float(np.median(arr)) if len(arr) else float("nan"),
            "p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
            "p99": float(np.percentile(arr, 99)) if len(arr) else float("nan"),
        }
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument(
        "--checkpoints", nargs="+",
        default=[
            "checkpoints/lepong_statehead_frozen.pt",
            "checkpoints/lepong_statehead_occ_aug.pt",
        ],
    )
    pa.add_argument("--occlusions", nargs="+", type=float,
                    default=[0.0, 0.2, 0.4, 0.6])
    pa.add_argument("--episodes", type=int, default=20)
    pa.add_argument("--steps", type=int, default=100)
    pa.add_argument("--frameskip", type=int, default=5)
    pa.add_argument("--seed", type=int, default=2349867)
    pa.add_argument("--output", default="results/eval_occlusion.json")
    args = pa.parse_args()

    device = torch.device("cpu")
    print("=== Occlusion A/B matrix ===", flush=True)
    print(f"  checkpoints: {args.checkpoints}", flush=True)
    print(f"  occlusions:  {args.occlusions}", flush=True)
    print(f"  eval size:   {args.episodes} episodes x {args.steps} steps", flush=True)

    # 1. Generate shared in-distribution frames once
    print("\nGenerating shared in-distribution frames (AI-tracked)...", flush=True)
    t0 = time.time()
    frames, states = generate_frames(
        n_episodes=args.episodes,
        steps=args.steps,
        frameskip=args.frameskip,
        seed=args.seed,
    )
    print(f"  {len(frames)} frames in {time.time() - t0:.1f}s", flush=True)

    # 2. For each occlusion level, generate occluded frames once
    occluded_frames = {}
    for occ in args.occlusions:
        occluded_frames[occ] = apply_occlusion(frames, occ)

    # 3. For each checkpoint, eval on each occlusion level
    matrix = {}
    for ckpt_path in args.checkpoints:
        ckpt_key = pathlib.Path(ckpt_path).stem
        print(f"\nLoading {ckpt_key}...", flush=True)
        model, state_mean, state_std = load_model(ckpt_path, device)
        matrix[ckpt_key] = {}
        for occ in args.occlusions:
            print(f"  evaluating on occ={occ:.1f}...", flush=True)
            t = time.time()
            errs = eval_state_head(
                occluded_frames[occ], states, model, state_mean, state_std, device
            )
            matrix[ckpt_key][f"occ_{occ:.1f}"] = summarize(errs)
            print(f"    done in {time.time() - t:.1f}s", flush=True)

    # 4. Print matrix
    print("\n" + "=" * 80, flush=True)
    print("MEDIAN ABSOLUTE ERROR MATRIX (lower is better)", flush=True)
    print("=" * 80, flush=True)
    for dim_idx in REPORT_DIMS:
        name = STATE_NAMES[dim_idx]
        print(f"\n{name}:", flush=True)
        header = "checkpoint".ljust(40) + " ".join(
            f"occ={o:.1f}".rjust(10) for o in args.occlusions
        )
        print(header, flush=True)
        for ckpt_key, occ_results in matrix.items():
            row = ckpt_key.ljust(40)
            for occ in args.occlusions:
                med = occ_results[f"occ_{occ:.1f}"][name]["median"]
                row += f"{med:10.4f}"
            print(row, flush=True)

    # 5. Save JSON
    results = {
        "checkpoints": args.checkpoints,
        "occlusions": args.occlusions,
        "episodes": args.episodes,
        "steps": args.steps,
        "matrix": matrix,
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.output}", flush=True)

    # 6. Key verdict
    print("\n=== VERDICT ===", flush=True)
    baseline_key = pathlib.Path(args.checkpoints[0]).stem
    aug_key = pathlib.Path(args.checkpoints[-1]).stem if len(args.checkpoints) > 1 else None
    if aug_key is None or aug_key == baseline_key:
        print("  Only one checkpoint tested -- no A/B.", flush=True)
    else:
        print(f"  {baseline_key} (baseline) vs {aug_key} (augmented):", flush=True)
        for dim_idx in REPORT_DIMS:
            name = STATE_NAMES[dim_idx]
            for occ in args.occlusions:
                base_med = matrix[baseline_key][f"occ_{occ:.1f}"][name]["median"]
                aug_med = matrix[aug_key][f"occ_{occ:.1f}"][name]["median"]
                delta = (aug_med - base_med) / max(base_med, 1e-6) * 100
                print(f"    {name} @ occ={occ:.1f}: "
                      f"{base_med:.3f} -> {aug_med:.3f}  ({delta:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
