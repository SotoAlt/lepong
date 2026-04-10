"""Train a frozen state head on Pong data.

Loads a pretrained JEPAPool checkpoint and freezes the encoder + predictor.
Only the Linear(192, 10) state head trains (~1,930 parameters).

Usage:
    python scripts/train_statehead.py \
        --data /path/to/pong_v1.npz \
        --init /path/to/lepong_v1.pt \
        --output /path/to/lepong_statehead_frozen.pt \
        --epochs 20 --batch 128 --lr 1e-3
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.jepa_pool import JEPAPool, EMBED_DIM, HISTORY_SIZE


class PongDataset(Dataset):
    """Yields (frames_window, actions_window, states_window) where window = HISTORY_SIZE+1."""
    def __init__(self, frames, actions, states, history=HISTORY_SIZE):
        self.frames = frames
        self.actions = actions
        self.states = states
        self.history = history
        self.indices = list(range(history, len(frames) - 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        f = self.frames[i - self.history:i + 1]
        a = self.actions[i - self.history:i + 1]
        s = self.states[i - self.history:i + 1]
        return f, a, s


def freeze_everything_except_state_head(model: JEPAPool) -> list[str]:
    """Flip requires_grad off on all modules except state_head.

    Returns the list of modules that were frozen, for logging and
    for saving in the checkpoint as an audit trail.
    """
    frozen_modules = []
    for p in model.encoder.parameters():
        p.requires_grad = False
    frozen_modules.append("encoder")
    for p in model.projector.parameters():
        p.requires_grad = False
    frozen_modules.append("projector")
    for p in model.action_encoder.parameters():
        p.requires_grad = False
    frozen_modules.append("action_encoder")
    for p in model.predictor.parameters():
        p.requires_grad = False
    frozen_modules.append("predictor")
    for p in model.pred_projector.parameters():
        p.requires_grad = False
    frozen_modules.append("pred_projector")
    for p in model.sigreg.parameters():
        p.requires_grad = False
    frozen_modules.append("sigreg")
    return frozen_modules


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data", required=True, help="Path to pong_v1.npz")
    pa.add_argument("--init", required=True, help="Init checkpoint (lepong_v1.pt)")
    pa.add_argument("--output", required=True, help="Output checkpoint path")
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--batch", type=int, default=128)
    pa.add_argument("--lr", type=float, default=1e-3,
                    help="Higher than full-model lr because only ~2K params train")
    pa.add_argument("--state-dim", type=int, default=10)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--num-workers", type=int, default=2)
    pa.add_argument("--val-frac", type=float, default=0.1)
    args = pa.parse_args()

    device = torch.device(args.device)
    print(f"=== train_statehead (frozen backbone) ===", flush=True)
    print(f"  device:  {device}", flush=True)
    print(f"  data:    {args.data}", flush=True)
    print(f"  init:    {args.init}", flush=True)
    print(f"  output:  {args.output}", flush=True)
    print(f"  epochs:  {args.epochs}", flush=True)
    print(f"  batch:   {args.batch}", flush=True)
    print(f"  lr:      {args.lr}", flush=True)
    print(f"  state_dim: {args.state_dim}", flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    d = np.load(args.data)
    print(f"  npz keys: {list(d.keys())}", flush=True)
    frames = torch.from_numpy(d["frames"]).float().permute(0, 3, 1, 2) / 255.0
    actions = torch.from_numpy(d["actions"]).float()
    states = torch.from_numpy(d["states"]).float()
    print(f"  frames: {frames.shape}, actions: {actions.shape}, states: {states.shape}", flush=True)
    actual_state_dim = states.shape[1]
    if actual_state_dim != args.state_dim:
        print(f"  NOTE: state_dim arg={args.state_dim}, actual={actual_state_dim}. Using actual.", flush=True)
        args.state_dim = actual_state_dim

    # Normalize states (per-dim)
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0).clamp(min=1e-6)
    states_norm = (states - state_mean) / state_std
    print(f"  state mean: {state_mean.tolist()}", flush=True)
    print(f"  state std:  {state_std.tolist()}", flush=True)

    # Train/val split
    n = len(frames)
    n_val = int(n * args.val_frac)
    n_train = n - n_val
    train_frames = frames[:n_train]
    train_actions = actions[:n_train]
    train_states = states_norm[:n_train]
    val_frames = frames[n_train:]
    val_actions = actions[n_train:]
    val_states = states_norm[n_train:]
    print(f"  train: {n_train}, val: {n_val}", flush=True)

    train_ds = PongDataset(train_frames, train_actions, train_states)
    val_ds = PongDataset(val_frames, val_actions, val_states)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  train windows: {len(train_ds)}, val windows: {len(val_ds)}", flush=True)

    # Model
    print(f"\nBuilding JEPAPool with state_dim={args.state_dim}...", flush=True)
    model = JEPAPool(state_dim=args.state_dim).to(device)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params_total:,}", flush=True)

    # Init from existing checkpoint (encoder + predictor already pretrained)
    print(f"\nLoading init checkpoint: {args.init}", flush=True)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    msg = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  missing: {len(msg.missing_keys)} keys (expected: state_head)", flush=True)
    print(f"  unexpected: {len(msg.unexpected_keys)} keys", flush=True)

    # Freeze the encoder + predictor + everything except the state head
    print("\nFreezing encoder + predictor + everything except state_head...", flush=True)
    frozen_modules = freeze_everything_except_state_head(model)
    for name in frozen_modules:
        print(f"  [frozen] model.{name}", flush=True)

    # Sanity check: count trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  trainable params: {n_trainable:,}  (should be ~2K for Linear(192, 10))", flush=True)
    print(f"  frozen params:    {n_frozen:,}  (should be ~13M)", flush=True)

    if n_trainable > 5000:
        print(f"\n  ERROR: expected ~1930 trainable params for a Linear(192, 10)"
              f" state head, got {n_trainable}. Something is wrong with the freezing.",
              flush=True)
        sys.exit(1)

    # Optimizer only sees trainable params
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3,
                            betas=(0.9, 0.95))

    state_names = ["ball_x", "ball_y", "ball_vx", "ball_vy", "pad_l", "pad_r",
                   "score_l", "score_r", "speed", "rally"]
    state_names = state_names[:args.state_dim]

    print("\n=== TRAINING (state head only) ===", flush=True)
    t0 = time.time()
    # Put the frozen modules into eval mode so BatchNorm running stats don't update
    model.encoder.eval()
    model.projector.eval()
    model.action_encoder.eval()
    model.predictor.eval()
    model.pred_projector.eval()

    for epoch in range(args.epochs):
        model.state_head.train()
        sum_pred, sum_state, n_b = 0.0, 0.0, 0

        for xb, ab, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)

            out = model(xb, ab, sb)
            pred_loss, _sigreg_loss, _, _, state_loss, _, _ = out
            total = state_loss

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            sum_pred += pred_loss.item() * len(xb)
            sum_state += state_loss.item() * len(xb)
            n_b += len(xb)

        avg_pred = sum_pred / n_b
        avg_state = sum_state / n_b

        # Validation
        model.state_head.eval()
        val_state_preds = []
        val_state_targets = []
        val_pred_sum = 0.0
        val_state_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, ab, sb in val_loader:
                xb = xb.to(device, non_blocking=True)
                ab = ab.to(device, non_blocking=True)
                sb = sb.to(device, non_blocking=True)
                out = model(xb, ab, sb)
                pred_loss, _, _, _, state_loss, state_pred, tgt_states = out
                val_pred_sum += pred_loss.item() * len(xb)
                val_state_sum += state_loss.item() * len(xb)
                val_n += len(xb)
                val_state_preds.append(state_pred.reshape(-1, args.state_dim).cpu())
                val_state_targets.append(tgt_states.reshape(-1, args.state_dim).cpu())

        val_pred_loss = val_pred_sum / val_n
        val_state_loss = val_state_sum / val_n
        all_p = torch.cat(val_state_preds)
        all_t = torch.cat(val_state_targets)
        corrs = []
        for i in range(args.state_dim):
            if all_t[:, i].std() < 1e-6:
                corrs.append(float("nan"))
            else:
                c = float(np.corrcoef(all_p[:, i].numpy(), all_t[:, i].numpy())[0, 1])
                corrs.append(c)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train: pred={avg_pred:.4f} (frozen, should be flat) state={avg_state:.4f}  "
              f"val: pred={val_pred_loss:.4f} state={val_state_loss:.4f}  "
              f"({elapsed:.0f}s)", flush=True)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            corr_str = " ".join(f"{n[:5]}={c:+.3f}" for n, c in zip(state_names, corrs))
            print(f"        val corrs: {corr_str}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TRAINING DONE in {elapsed:.0f}s ===", flush=True)

    # Final eval
    print("\nFinal val correlations (state head reading frozen encoder + predictor):",
          flush=True)
    for i, name in enumerate(state_names):
        c = corrs[i]
        status = "GOOD" if c > 0.85 else ("OK" if c > 0.5 else "WEAK")
        print(f"  {name:10s}: {c:+.4f} [{status}]", flush=True)

    # Save checkpoint
    save_dict = {
        "model": model.state_dict(),
        "embed_dim": EMBED_DIM,
        "state_dim": args.state_dim,
        "state_mean": state_mean,
        "state_std": state_std,
        "state_names": state_names,
        "val_correlations": dict(zip(state_names, corrs)),
        "epochs": args.epochs,
        "lr": args.lr,
        "frozen_modules": frozen_modules,
        "trainable_param_count": n_trainable,
        "frozen_param_count": n_frozen,
        "init_checkpoint": args.init,
        "training_script": "scripts/train_statehead.py",
    }
    torch.save(save_dict, args.output)
    print(f"\nSaved checkpoint: {args.output}", flush=True)
    print(f"  size: {pathlib.Path(args.output).stat().st_size / 1e6:.1f} MB", flush=True)
    print(f"  frozen modules: {frozen_modules}", flush=True)


if __name__ == "__main__":
    main()
