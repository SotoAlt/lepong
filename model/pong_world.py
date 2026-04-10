"""Pong physics engine.

Smooth linear ball dynamics with wall/paddle bounces.
The ball trajectory between bounces is linear, making it
well-suited for JEPA prediction (only 10 state floats).

Usage:
    env = PongWorld()
    env.reset()
    for _ in range(100):
        action = [left_paddle_dy, right_paddle_dy]
        env.step(action)
        img = env.render()  # (128, 128, 3) uint8
"""
from __future__ import annotations

import math
import numpy as np


# Court dimensions (normalized)
COURT_W = 1.0
COURT_H = 0.6
PADDLE_H = 0.12
PADDLE_W = 0.015
PADDLE_MARGIN = 0.03  # distance from wall
BALL_R = 0.012
BALL_SPEED_MIN = 0.015
BALL_SPEED_MAX = 0.025
PADDLE_SPEED = 0.02


class PongWorld:
    def __init__(self):
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.paddle_l = 0.0  # y center of left paddle
        self.paddle_r = 0.0  # y center of right paddle
        self.score_l = 0
        self.score_r = 0
        self.rally = 0
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.ball_x = COURT_W / 2
        self.ball_y = COURT_H / 2
        speed = self.rng.uniform(BALL_SPEED_MIN, BALL_SPEED_MAX)
        angle = self.rng.uniform(-0.4, 0.4)  # slight vertical angle
        direction = self.rng.choice([-1, 1])
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)
        self.paddle_l = COURT_H / 2
        self.paddle_r = COURT_H / 2
        self.score_l = 0
        self.score_r = 0
        self.rally = 0
        return self.get_state()

    def step(self, action=None):
        """action: [left_paddle_dy, right_paddle_dy] each in [-1, 1]."""
        if action is not None:
            dl = float(action[0]) * PADDLE_SPEED
            dr = float(action[1]) * PADDLE_SPEED
        else:
            dl = dr = 0.0

        # Move paddles (clamp to court)
        half_h = PADDLE_H / 2
        self.paddle_l = np.clip(self.paddle_l + dl, half_h, COURT_H - half_h)
        self.paddle_r = np.clip(self.paddle_r + dr, half_h, COURT_H - half_h)

        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Top/bottom wall bounce
        if self.ball_y - BALL_R < 0:
            self.ball_y = BALL_R
            self.ball_vy = abs(self.ball_vy)
        if self.ball_y + BALL_R > COURT_H:
            self.ball_y = COURT_H - BALL_R
            self.ball_vy = -abs(self.ball_vy)

        # Left paddle collision
        lx = PADDLE_MARGIN + PADDLE_W
        if (self.ball_x - BALL_R < lx and self.ball_vx < 0 and
                abs(self.ball_y - self.paddle_l) < half_h + BALL_R):
            self.ball_x = lx + BALL_R
            self.ball_vx = abs(self.ball_vx) * 1.02  # slight speedup
            # Angle adjustment based on hit position
            offset = (self.ball_y - self.paddle_l) / half_h  # -1 to 1
            self.ball_vy += offset * 0.005
            self.rally += 1

        # Right paddle collision
        rx = COURT_W - PADDLE_MARGIN - PADDLE_W
        if (self.ball_x + BALL_R > rx and self.ball_vx > 0 and
                abs(self.ball_y - self.paddle_r) < half_h + BALL_R):
            self.ball_x = rx - BALL_R
            self.ball_vx = -abs(self.ball_vx) * 1.02
            offset = (self.ball_y - self.paddle_r) / half_h
            self.ball_vy += offset * 0.005
            self.rally += 1

        # Scoring (ball passes paddle)
        scored = False
        if self.ball_x < 0:
            self.score_r += 1
            scored = True
        if self.ball_x > COURT_W:
            self.score_l += 1
            scored = True

        if scored:
            # Reset ball from center
            self.ball_x = COURT_W / 2
            self.ball_y = COURT_H / 2
            speed = self.rng.uniform(BALL_SPEED_MIN, BALL_SPEED_MAX)
            angle = self.rng.uniform(-0.4, 0.4)
            direction = self.rng.choice([-1, 1])
            self.ball_vx = direction * speed * math.cos(angle)
            self.ball_vy = speed * math.sin(angle)
            self.rally = 0

        # Cap ball speed
        speed = math.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if speed > BALL_SPEED_MAX * 1.5:
            scale = BALL_SPEED_MAX * 1.5 / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """10-float state vector."""
        speed = math.sqrt(self.ball_vx**2 + self.ball_vy**2)
        return np.array([
            self.ball_x / COURT_W,
            self.ball_y / COURT_H,
            self.ball_vx / BALL_SPEED_MAX,
            self.ball_vy / BALL_SPEED_MAX,
            self.paddle_l / COURT_H,
            self.paddle_r / COURT_H,
            min(self.score_l, 10) / 10.0,
            min(self.score_r, 10) / 10.0,
            speed / BALL_SPEED_MAX,
            min(self.rally, 20) / 20.0,
        ], dtype=np.float32)

    def render(self, size: int = 128) -> np.ndarray:
        """Render as (size, size, 3) uint8 RGB."""
        # Aspect ratio: court is wider than tall
        w = size
        h = int(size * COURT_H / COURT_W)
        y_off = (size - h) // 2
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Court background
        img[y_off:y_off + h, :] = (20, 20, 30)

        # Center line (dashed)
        cx = size // 2
        for y in range(y_off, y_off + h, 6):
            img[y:y+3, cx-1:cx+1] = (60, 60, 70)

        # Top/bottom borders
        img[y_off:y_off+2, :] = (100, 100, 110)
        img[y_off+h-2:y_off+h, :] = (100, 100, 110)

        def to_px(nx, ny):
            return int(nx * w), y_off + int(ny * h / COURT_H)

        # Paddles
        def draw_paddle(px_x, center_y, color):
            pw = max(2, int(PADDLE_W * w))
            ph = max(4, int(PADDLE_H * h / COURT_H))
            py = y_off + int(center_y * h / COURT_H) - ph // 2
            x0 = max(0, px_x - pw // 2)
            x1 = min(size, px_x + pw // 2)
            y0 = max(y_off, py)
            y1 = min(y_off + h, py + ph)
            img[y0:y1, x0:x1] = color

        lx = int(PADDLE_MARGIN * w)
        rx = int((COURT_W - PADDLE_MARGIN) * w)
        draw_paddle(lx, self.paddle_l, (200, 200, 220))
        draw_paddle(rx, self.paddle_r, (200, 200, 220))

        # Ball
        bx, by = to_px(self.ball_x, self.ball_y)
        br = max(2, int(BALL_R * w))
        for dy in range(-br, br + 1):
            for dx in range(-br, br + 1):
                if dx*dx + dy*dy <= br*br:
                    px, py = bx + dx, by + dy
                    if 0 <= px < size and 0 <= py < size:
                        img[py, px] = (240, 240, 255)

        # Score (simple pixel numbers at top)
        # Left score
        sx = size // 4
        sy = y_off + 8
        for i in range(min(self.score_l, 5)):
            img[sy:sy+3, sx + i*5:sx + i*5 + 3] = (150, 150, 160)
        # Right score
        sx = 3 * size // 4
        for i in range(min(self.score_r, 5)):
            img[sy:sy+3, sx + i*5:sx + i*5 + 3] = (150, 150, 160)

        return img

    def ai_action(self, noise=0.0) -> list[float]:
        """Simple AI: both paddles track ball y with optional noise."""
        target_l = self.ball_y + self.rng.uniform(-noise, noise)
        target_r = self.ball_y + self.rng.uniform(-noise, noise)
        dl = np.clip((target_l - self.paddle_l) / PADDLE_SPEED, -1, 1)
        dr = np.clip((target_r - self.paddle_r) / PADDLE_SPEED, -1, 1)
        return [float(dl), float(dr)]


def generate_dataset(n_episodes=1000, steps_per_ep=100, frameskip=5, seed=42):
    """Generate Pong training data with AI players."""
    env = PongWorld()
    rng = np.random.default_rng(seed)

    all_frames, all_states, all_actions, all_episodes = [], [], [], []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        noise = rng.uniform(0.0, 0.15)  # varying AI skill per episode

        for step in range(steps_per_ep * frameskip):
            action = env.ai_action(noise=noise)
            env.step(action)

            if step % frameskip == 0:
                frame = env.render(128)
                state = env.get_state()
                all_frames.append(frame)
                all_states.append(state)
                all_actions.append(np.array(action, dtype=np.float32))
                all_episodes.append(ep)

        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes} "
                  f"(noise={noise:.2f}, score={env.score_l}-{env.score_r})")

    return {
        "frames": np.array(all_frames),
        "states": np.array(all_states),
        "actions": np.array(all_actions),
        "episodes": np.array(all_episodes),
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes", type=int, default=1000)
    pa.add_argument("--steps", type=int, default=100)
    pa.add_argument("--output", default="data/pong_v1.npz")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--preview", action="store_true")
    args = pa.parse_args()

    if args.preview:
        from PIL import Image
        env = PongWorld()
        env.reset(seed=42)
        for i in range(80):
            action = env.ai_action(noise=0.05)
            env.step(action)
            if i % 20 == 0:
                img = env.render(256)
                Image.fromarray(img).save(f"/tmp/pong_step_{i}.png")
                print(f"Saved /tmp/pong_step_{i}.png "
                      f"ball=({env.ball_x:.2f},{env.ball_y:.2f}) "
                      f"score={env.score_l}-{env.score_r}")
    else:
        print(f"Generating {args.episodes} Pong episodes x {args.steps} steps "
              f"(frameskip=5)...")
        data = generate_dataset(args.episodes, args.steps, seed=args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **data)
        print(f"Saved {len(data['frames'])} frames to {args.output}")
        for k in data:
            print(f"  {k}: {data[k].shape}")
