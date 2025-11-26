#!/usr/bin/env python3
"""
inference_refine_plus.py

Refine dance sequences with a trained RecurrentPPO (MlpLstmPolicy) model.
Outputs the refined .pkl in the same format as trainingv0.5:
 - smpl_poses: (T,72)
 - smpl_trans: (T,3)
 - full_pose:  (T,24,3)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import librosa
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

class DanceRefineEnv(gym.Env):
    """
    Dance refinement environment for inference.
    Observations and actions match training.
    """

    metadata = {"render.modes": ["none"]}

    def __init__(
        self,
        poses_72: np.ndarray,       # (T,72)
        trans: np.ndarray,          # (T,3)
        full_pose: np.ndarray,      # (T,24,3)
        beat_times: np.ndarray,
        audio_energy: np.ndarray,
        fps: int = 30,
        action_scale: float = 0.01,
        max_disp: float = 0.12,
    ):
        super().__init__()
        assert poses_72.ndim == 2 and poses_72.shape[1] == 72
        assert full_pose.ndim == 3 and full_pose.shape[1:] == (24, 3)

        self.poses_orig = poses_72
        self.trans_orig = trans
        self.full_poses_orig = full_pose
        self.beat_times = beat_times
        self.audio_energy_all = audio_energy
        self.fps = fps

        self.T = poses_72.shape[0]
        self.pose_dim = poses_72.shape[1]
        self.action_scale = action_scale
        self.max_disp = max_disp

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.pose_dim,), dtype=np.float32)
        obs_dim = self.pose_dim * 2 + 2
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_dim,), dtype=np.float32)

        self.modified = None
        self.full_modified = None
        self.t = 0
        self.prev_pose = np.zeros((self.pose_dim,), dtype=np.float32)
        self.current_pose = None

    def _beat_proximity(self, t: int) -> float:
        if len(self.beat_times) == 0:
            return 0.0
        abs_time = t / self.fps
        closest = self.beat_times[np.argmin(np.abs(self.beat_times - abs_time))]
        return float(np.tanh((closest - abs_time) * 5.0))

    def _audio_energy_at(self, t: int) -> float:
        return float(self.audio_energy_all[t]) if 0 <= t < len(self.audio_energy_all) else 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.modified = self.poses_orig.copy()
        self.full_modified = self.full_poses_orig.copy()
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

            disp = self.modified[next_idx] - self.modified[self.t]
            norm = np.linalg.norm(disp)
            if norm > self.max_disp:
                self.modified[next_idx] = self.modified[self.t] + disp * (self.max_disp / (norm + 1e-8))

        self.prev_pose = self.current_pose.copy()
        self.t += 1
        done = self.t >= self.T - 1
        if not done:
            self.current_pose = self.modified[self.t].copy()
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, done, False, {}

    def get_refined(self):
        """Return refined (T,72) and full_pose (T,24,3)."""
        return self.modified.copy(), self.trans_orig.copy(), self.full_modified.copy()



def load_pair(pkl_path: Path, wav_path: Path, sr: int = 22050):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    poses = np.array(data["smpl_poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"{pkl_path}: smpl_poses must be (T,72), got {poses.shape}")

    trans = np.array(data.get("smpl_trans", np.zeros((poses.shape[0], 3), dtype=np.float32)), dtype=np.float32)

    full_key = "full_pose"
    if full_key in data:
        full_poses = np.array(data[full_key], dtype=np.float32)
    else:
        full_poses = poses.reshape(poses.shape[0], 24, 3)

    L = min(poses.shape[0], trans.shape[0], full_poses.shape[0])
    poses, trans, full_poses = poses[:L], trans[:L], full_poses[:L]

    if not wav_path.exists():
        beat_times = np.array([], dtype=np.float32)
        audio_energy = np.ones((L,), dtype=np.float32) * 0.1
    else:
        y, sr_loaded = librosa.load(str(wav_path), sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_loaded)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr_loaded)
        rms = librosa.feature.rms(y=y)[0]
        times_rms = np.linspace(0, len(y) / sr_loaded, num=len(rms))
        times_pose = np.linspace(0, len(y) / sr_loaded, num=L)
        audio_energy = np.interp(times_pose, times_rms, rms)

    return poses, trans, full_poses, beat_times, audio_energy




def motion_stats(poses_72):
    # Compute mean velocity and acceleration for (T,72) sequences.
    if poses_72.shape[0] < 3:
        return 0.0, 0.0
    vel = np.linalg.norm(np.diff(poses_72, axis=0), axis=-1)
    acc = np.linalg.norm(np.diff(poses_72, n=2, axis=0), axis=-1)
    return vel.mean(), acc.mean()



def refine_single(model_path, input_pkl, output_pkl, audio_dir,
                  vecnorm_path=None, deterministic=True, temperature=1.0, device="cpu"):

    pkl_path = Path(input_pkl)
    wav_path = Path(audio_dir) / (pkl_path.stem + ".wav")


    print(f"[+] Loading data: {pkl_path.name}")
    poses, trans, full_poses, beats, audio_e = load_pair(pkl_path, wav_path)
    env = DanceRefineEnv(poses, trans, full_poses, beats, audio_e)


    print(f"[+] Loading model from {model_path}")
    model = RecurrentPPO.load(model_path, device=device)


    if vecnorm_path and Path(vecnorm_path).exists():
        print(f"[+] Loading VecNormalize stats from {vecnorm_path}")
        vecnorm = VecNormalize.load(vecnorm_path, env)
        env = vecnorm.venv
    else:
        print("[!] VecNormalize stats not found or skipped; results may differ slightly.")

    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    print("[+] Running refinement...")
    for _ in range(env.T - 1):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=deterministic
        )
        if not deterministic and temperature != 1.0:
            action = action + np.random.normal(0, (1 - temperature) * 0.05, size=action.shape)

        obs, _, done, _, _ = env.step(action)
        episode_starts = np.array([done])
        if done:
            break

    refined_poses, refined_trans, refined_full = env.get_refined()


    vel_b, acc_b = motion_stats(poses)
    vel_a, acc_a = motion_stats(refined_poses)
    print(f"Velocity mean: before={vel_b:.5f}, after={vel_a:.5f}")
    print(f"Acceleration mean: before={acc_b:.5f}, after={acc_a:.5f}")
    print(f"Energy ratio (after/before): {vel_a / (vel_b + 1e-8):.3f}")

    # --- Save output ---
    output_data = {
        "smpl_poses": refined_poses,   # (T,72)
        "smpl_trans": refined_trans,   # (T,3)
        "full_pose": refined_full,     # (T,24,3)
        "audio_energy": audio_e,
        "refined": True,
        "vel_mean_before": vel_b,
        "vel_mean_after": vel_a,
        "acc_mean_before": acc_b,
        "acc_mean_after": acc_a,
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(output_data, f)

    print(f"[✓] Refined sequence saved → {output_pkl}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to .zip model (e.g., rppo_dance_final.zip)")
    p.add_argument("--input", type=str, required=True, help="Input .pkl file to refine")
    p.add_argument("--output", type=str, required=True, help="Output refined .pkl path")
    p.add_argument("--audio_dir", type=str, required=True, help="Directory containing matching .wav files")
    p.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize pickle (optional)")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic inference")
    p.add_argument("--temperature", type=float, default=1.0, help="Noise scale for stochastic refinement (1=no noise)")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = p.parse_args()

    refine_single(
        model_path=args.model,
        input_pkl=args.input,
        output_pkl=args.output,
        audio_dir=args.audio_dir,
        vecnorm_path=args.vecnorm,
        deterministic=args.deterministic,
        temperature=args.temperature,
        device=args.device,
    )
