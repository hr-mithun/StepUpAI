
import argparse
import os
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import librosa
import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def load_pair(pkl_path: Path, wav_path: Path, sr: int = 22050):

       #smpl_poses -> (T,72)
       #smpl_trans -> (T,3)
       #full_pose  -> (T,24,3)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "smpl_poses" not in data:
        raise KeyError(f"{pkl_path} missing 'smpl_poses'")

    poses = np.array(data["smpl_poses"], dtype=np.float32)  # (T,72)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"{pkl_path}: smpl_poses must be (T,72), got {poses.shape}")

    trans = np.array(
        data.get("smpl_trans", np.zeros((poses.shape[0], 3), dtype=np.float32)),
        dtype=np.float32,
    )

    full_key = "full_pose"
    if full_key in data:
        full_poses = np.array(data[full_key], dtype=np.float32)
        if full_poses.ndim != 3 or full_poses.shape[1:] != (24, 3):
            raise ValueError(f"{pkl_path}: full_pose must be (T,24,3), got {full_poses.shape}")
    else:
        full_poses = poses.reshape(poses.shape[0], 24, 3)

    L = min(poses.shape[0], trans.shape[0], full_poses.shape[0])
    poses, trans, full_poses = poses[:L], trans[:L], full_poses[:L]

    # Add translation to root (first 3 of flattened pose)
    poses[:, :3] += trans
    root_init = poses[0, :3].copy()
    poses_centered = poses.copy()
    poses_centered[:, :3] -= root_init

    # full_pose also shifted by same translation
    full_poses_centered = full_poses.copy()
    full_poses_centered[:, 0, :] += trans
    full_poses_centered -= full_poses_centered[0, 0:1, :]

    poses_norm = poses_centered.astype(np.float32)
    full_poses_norm = full_poses_centered.astype(np.float32)
    scale = 1.0

    # --- Audio ---
    if not wav_path.exists():
        beat_times = np.array([], dtype=np.float32)
        audio_energy = np.ones((poses_norm.shape[0],), dtype=np.float32) * 0.1
    else:
        y, sr_loaded = librosa.load(str(wav_path), sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_loaded)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr_loaded)
        rms = librosa.feature.rms(y=y)[0]
        times_rms = np.linspace(0, len(y) / sr_loaded, num=len(rms))
        times_pose = np.linspace(0, len(y) / sr_loaded, num=poses_norm.shape[0])
        audio_energy = np.interp(times_pose, times_rms, rms)

    return (
        poses_norm,        # (T,72)
        trans.astype(np.float32),
        full_poses_norm,   # (T,24,3)
        beat_times.astype(np.float32),
        audio_energy.astype(np.float32),
        scale,
        pkl_path.stem,
    )


def load_dataset_pairs(data_dir: Path, sr: int = 22050):
    files = sorted(list(data_dir.glob("**/*.pkl")))
    print(f"Found {len(files)} .pkl files.")
    dataset = []
    for p in files:
        wav = data_dir / (p.stem + ".wav")
        if not wav.exists():
            print(f"[load_dataset_pairs] Skipping {p.name}: missing {wav.name}")
            continue
        dataset.append(load_pair(p, wav, sr=sr))
    if not dataset:
        raise RuntimeError("No valid .pkl + .wav pairs found in data_dir")
    return dataset



class DanceRefineEnv(gym.Env):
    metadata = {"render.modes": ["none"]}

    def __init__(
        self,
        poses: np.ndarray,            # (T,72)
        trans: np.ndarray,            # (T,3)
        full_poses: np.ndarray,       # (T,24,3)
        beat_times: np.ndarray,
        audio_energy: np.ndarray,
        fps=30,
        action_scale=0.01,
        max_disp=0.12,
        subseq_len=None,
        basename="",
        w_smooth=0.25,
        w_energy=0.6,
        w_beat=0.6,
        w_anchor=1.0,
        w_realism=0.05,
        anchor_strength=3.0,
        energy_gain=8.0,
        realism_max_joint_radius=3.0,
    ):
        super().__init__()

        assert poses.ndim == 2 and poses.shape[1] == 72
        assert full_poses.ndim == 3 and full_poses.shape[1:] == (24, 3)

        self.poses_orig = poses
        self.trans_orig = trans
        self.full_poses_orig = full_poses
        self.beat_times = beat_times
        self.audio_energy_all = audio_energy
        self.fps = fps
        self.T = poses.shape[0]
        self.pose_dim = poses.shape[1]

        self.fixed_subseq_len = subseq_len
        self.basename = basename
        self.action_scale = action_scale
        self.max_disp = max_disp

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.pose_dim,), dtype=np.float32)
        obs_dim = self.pose_dim * 2 + 2
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_dim,), dtype=np.float32)

        self.w_smooth = w_smooth
        self.w_energy = w_energy
        self.w_beat = w_beat
        self.w_anchor = w_anchor
        self.w_realism = w_realism
        self.anchor_strength = anchor_strength
        self.energy_gain = energy_gain
        self.realism_max_joint_radius = realism_max_joint_radius

        self.modified = None
        self.t = 0
        self.prev_pose = np.zeros((self.pose_dim,), dtype=np.float32)
        self.current_pose = None

    def _beat_proximity(self, t):
        if len(self.beat_times) == 0:
            return 0.0
        abs_time = t / self.fps
        closest = self.beat_times[np.argmin(np.abs(self.beat_times - abs_time))]
        return float(np.tanh((closest - abs_time) * 5.0))

    def _audio_energy_at(self, t):
        return float(self.audio_energy_all[t]) if 0 <= t < len(self.audio_energy_all) else 0.0

    def compute_reward(self, prev_pose, cur_pose, nxt_pose, audio_e, beat_prox):
        acc = np.mean(np.abs((nxt_pose - cur_pose) - (cur_pose - prev_pose)))
        r_smooth = -acc

        vel_next = np.mean(np.abs(nxt_pose - cur_pose))
        target = audio_e * self.energy_gain
        r_energy = -abs(vel_next - target) / (target + 1e-8)

        r_beat = np.exp(-abs(beat_prox) * 3.0) - 0.5
        orig_next = self.poses_orig[min(self.T - 1, self.t + 1)]
        mean_disp = np.mean(np.abs(nxt_pose - orig_next))
        r_anchor = -self.anchor_strength * mean_disp

        excess = np.clip(np.abs(nxt_pose - self.poses_orig[self.t]) - self.realism_max_joint_radius, 0, None)
        r_realism = -np.sum(excess)

        total = (
            self.w_smooth * r_smooth
            + self.w_energy * r_energy
            + self.w_beat * r_beat
            + self.w_anchor * r_anchor
            + self.w_realism * r_realism
        )
        return float(np.clip(total, -10, 10))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.modified = self.poses_orig.copy()
        self.t = 0
        self.prev_pose = np.zeros_like(self.modified[0])
        self.current_pose = self.modified[0].copy()
        return self._get_obs(), {}

    def _get_obs(self):
        audio_e = self._audio_energy_at(self.t)
        beat_prox = self._beat_proximity(self.t)
        return np.concatenate([
            self.current_pose,
            self.prev_pose,
            np.array([audio_e, beat_prox], dtype=np.float32)
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.action_scale
        reward, done = 0.0, False

        if self.t < self.T - 1:
            next_idx = self.t + 1
            self.modified[next_idx] += delta
            audio_e = self._audio_energy_at(self.t)
            beat_prox = self._beat_proximity(self.t)
            reward = self.compute_reward(self.prev_pose, self.current_pose, self.modified[next_idx], audio_e, beat_prox)

        self.prev_pose = self.current_pose.copy()
        self.t += 1
        done = self.t >= self.T - 1
        if not done:
            self.current_pose = self.modified[self.t].copy()
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, done, False, {}



def make_vec_envs(dataset, num_envs=2, fps=30, clip_obs=10.0, norm_reward=True):
    def make_one():
        poses, trans, full_poses, beats, audio_energy, scale, basename = dataset[np.random.randint(len(dataset))]
        return DanceRefineEnv(poses, trans, full_poses, beats, audio_energy, fps=fps)
    env_fns = [make_one for _ in range(num_envs)]
    vec = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs)
    return vec


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_pairs(data_dir)
    print(f"[main] Loaded {len(dataset)} sequences.")

    vec_env = make_vec_envs(dataset, num_envs=args.num_envs, fps=args.fps)
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])], lstm_hidden_size=128, n_lstm_layers=1, shared_lstm=False)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        device=device,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
    )

    eval_cb = EvalCallback(vec_env, best_model_save_path=str(out_dir / "best_model"), log_path=str(out_dir / "eval_logs"), eval_freq=max(1, args.eval_freq // args.num_envs), deterministic=True)
    ckpt_cb = CheckpointCallback(save_freq=max(1, args.save_freq // args.num_envs), save_path=str(out_dir / "checkpoints"), name_prefix="rppo")

    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, eval_cb])
    model.save(str(out_dir / "rppo_dance_final.zip"))
    print(f"[main] Training complete — model saved to {out_dir}/rppo_dance_final.zip")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./ppo_out")
    p.add_argument("--total_timesteps", type=int, default=50000)
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.25)
    p.add_argument("--ent_coef", type=float, default=0.25)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--save_freq", type=int, default=10000)
    p.add_argument("--eval_freq", type=int, default=10000)
    args = p.parse_args()
    main(args)
