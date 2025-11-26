"""
NFR checks for RL dance refinement project
Performance • Reproducibility • Stability
Run with:  python nfr_checks.py
"""

import time
import pickle
from pathlib import Path

import numpy as np

import trainingv0_5 as train_mod
import infer5


# ---------- Helpers ----------

def make_dummy_seq(T=300):
    """Create a dummy (T,72)/(T,3)/(T,24,3) triple."""
    poses = np.zeros((T, 72), dtype=np.float32)
    trans = np.zeros((T, 3), dtype=np.float32)
    full_pose = np.zeros((T, 24, 3), dtype=np.float32)
    return poses, trans, full_pose


def banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------- PERFORMANCE ----------

def check_performance_training_episode(T: int = 800, time_threshold: float = 2.0):
    banner("PERFORMANCE: Training environment episode")

    poses, trans, full_pose = make_dummy_seq(T)
    beat_times = np.array([], dtype=np.float32)
    audio_energy = np.ones((T,), dtype=np.float32) * 0.1

    env = train_mod.DanceRefineEnv(
        poses=poses,
        trans=trans,
        full_poses=full_pose,
        beat_times=beat_times,
        audio_energy=audio_energy,
        fps=30,
    )

    obs, _ = env.reset()

    start = time.perf_counter()
    done = False
    steps = 0
    while not done:
        action = np.zeros((env.pose_dim,), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    elapsed = time.perf_counter() - start

    print(f"T = {T}, steps = {steps}")
    print(f"Episode time: {elapsed:.4f} s  (threshold: {time_threshold:.2f} s)")
    print("Result:", "PASS" if elapsed < time_threshold else "FAIL")
    return elapsed


def check_performance_inference_dummy_env(T: int = 700, time_threshold: float = 3.0):
    """
    Performance check using the inference DanceRefineEnv directly
    (does NOT require a trained model or actual files).
    """
    banner("PERFORMANCE: Inference environment rollout (no model)")

    poses, trans, full_pose = make_dummy_seq(T)
    beat_times = np.array([], dtype=np.float32)
    audio_energy = np.ones((T,), dtype=np.float32) * 0.1

    env = infer5.DanceRefineEnv(
        poses_72=poses,
        trans=trans,
        full_pose=full_pose,
        beat_times=beat_times,
        audio_energy=audio_energy,
        fps=30,
    )

    obs, _ = env.reset()
    start = time.perf_counter()
    done = False
    steps = 0
    while not done:
        action = np.zeros((env.pose_dim,), dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    elapsed = time.perf_counter() - start

    print(f"T = {T}, steps = {steps}")
    print(f"Inference env rollout time: {elapsed:.4f} s  (threshold: {time_threshold:.2f} s)")
    print("Result:", "PASS" if elapsed < time_threshold else "FAIL")
    return elapsed


# ---------- REPRODUCIBILITY ----------

def check_reproducibility_inference_with_model(
    model_path: str,
    input_pkl: str,
    audio_dir: str,
    vecnorm_path: str | None = None,
):
    """
    Runs refine_single twice with deterministic=True and compares outputs.
    Use your REAL trained model + real input .pkl here.
    """
    banner("REPRODUCIBILITY: refine_single deterministic run x2")

    model_path = str(Path(model_path))
    input_pkl = str(Path(input_pkl))
    audio_dir = str(Path(audio_dir))

    tmp_dir = Path("nfr_outputs")
    tmp_dir.mkdir(exist_ok=True)
    out1 = tmp_dir / "refined_1.pkl"
    out2 = tmp_dir / "refined_2.pkl"

    # First run
    start1 = time.perf_counter()
    infer5.refine_single(
        model_path=model_path,
        input_pkl=input_pkl,
        output_pkl=str(out1),
        audio_dir=audio_dir,
        vecnorm_path=vecnorm_path,
        deterministic=True,
        temperature=1.0,
        device="cpu",
    )
    t1 = time.perf_counter() - start1

    # Second run
    start2 = time.perf_counter()
    infer5.refine_single(
        model_path=model_path,
        input_pkl=input_pkl,
        output_pkl=str(out2),
        audio_dir=audio_dir,
        vecnorm_path=vecnorm_path,
        deterministic=True,
        temperature=1.0,
        device="cpu",
    )
    t2 = time.perf_counter() - start2

    with out1.open("rb") as f1, out2.open("rb") as f2:
        d1 = pickle.load(f1)
        d2 = pickle.load(f2)

    poses1 = d1["smpl_poses"]
    poses2 = d2["smpl_poses"]

    equal = np.array_equal(poses1, poses2)
    max_diff = np.max(np.abs(poses1 - poses2)) if poses1.shape == poses2.shape else np.inf

    print(f"Run 1 time: {t1:.4f} s")
    print(f"Run 2 time: {t2:.4f} s")
    print(f"Shapes equal: {poses1.shape == poses2.shape}")
    print(f"Exact equality: {equal}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print("Result:", "PASS (deterministic)" if equal else "WARN (small numerical noise ok)")


# ---------- STABILITY ----------

def check_stability_training_random_actions(T: int = 200):
    banner("STABILITY: Training env with random actions")

    poses, trans, full_pose = make_dummy_seq(T)
    beat_times = np.array([], dtype=np.float32)
    audio_energy = np.ones((T,), dtype=np.float32) * 0.1

    env = train_mod.DanceRefineEnv(
        poses=poses,
        trans=trans,
        full_poses=full_pose,
        beat_times=beat_times,
        audio_energy=audio_energy,
        fps=30,
    )

    obs, _ = env.reset()
    rng = np.random.default_rng(42)

    done = False
    steps = 0
    while not done:
        action = rng.uniform(-1.0, 1.0, size=(env.pose_dim,)).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

        if not np.isfinite(reward) or not np.all(np.isfinite(obs)):
            print("FAIL: Found NaN/Inf in training env.")
            return

    print(f"Steps: {steps}")
    print("All rewards/observations finite. Result: PASS")


def check_stability_infer_random_actions(T: int = 200):
    banner("STABILITY: Inference env with random actions")

    poses, trans, full_pose = make_dummy_seq(T)
    beat_times = np.array([], dtype=np.float32)
    audio_energy = np.ones((T,), dtype=np.float32) * 0.1

    env = infer5.DanceRefineEnv(
        poses_72=poses,
        trans=trans,
        full_pose=full_pose,
        beat_times=beat_times,
        audio_energy=audio_energy,
        fps=30,
    )

    obs, _ = env.reset()
    rng = np.random.default_rng(123)

    done = False
    steps = 0
    while not done:
        action = rng.uniform(-1.0, 1.0, size=(env.pose_dim,)).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

        if not np.isfinite(reward) or not np.all(np.isfinite(obs)):
            print("FAIL: Found NaN/Inf in inference env.")
            return

    refined_poses, refined_trans, refined_full = env.get_refined()
    if not (np.all(np.isfinite(refined_poses))
            and np.all(np.isfinite(refined_trans))
            and np.all(np.isfinite(refined_full))):
        print("FAIL: NaN/Inf in refined outputs.")
    else:
        print(f"Steps: {steps}")
        print("All refined outputs finite. Result: PASS")


def check_stability_missing_audio(tmp_dir: Path | None = None):
    banner("STABILITY: Handling missing audio file")

    if tmp_dir is None:
        tmp_dir = Path("nfr_tmp")
    tmp_dir.mkdir(exist_ok=True)

    T = 50
    poses, trans, full_pose = make_dummy_seq(T)
    pkl_path = tmp_dir / "seq.pkl"
    wav_path = tmp_dir / "seq.wav"  # we will NOT create this -> missing audio

    with pkl_path.open("wb") as f:
        pickle.dump(
            {"smpl_poses": poses, "smpl_trans": trans, "full_pose": full_pose},
            f,
        )

    # Training loader
    poses_t, trans_t, full_t, beats_t, audio_e_t, scale_t, stem_t = train_mod.load_pair(
        pkl_path, wav_path
    )

    # Inference loader
    poses_i, trans_i, full_i, beats_i, audio_e_i = infer5.load_pair(
        pkl_path, wav_path
    )

    print(f"Training loader: beats={beats_t.shape[0]}, audio_len={audio_e_t.shape[0]}")
    print(f"Infer loader:    beats={beats_i.shape[0]}, audio_len={audio_e_i.shape[0]}")

    if (beats_t.shape[0] == 0 and beats_i.shape[0] == 0
            and audio_e_t.shape[0] == T and audio_e_i.shape[0] == T):
        print("Missing-audio handling looks stable. Result: PASS")
    else:
        print("Result: WARN/FAIL – unexpected shapes in missing-audio path.")


# ---------- MAIN ----------

def main():
    # 1) Performance
    check_performance_training_episode()
    check_performance_inference_dummy_env()

    # 2) Stability (purely synthetic)
    check_stability_training_random_actions()
    check_stability_infer_random_actions()
    check_stability_missing_audio()

    # 3) Reproducibility (requires actual model + real input pkl)
    #    Uncomment and fill paths when you want to test it.
    #
    # check_reproducibility_inference_with_model(
    #     model_path="path/to/your_trained_model.zip",
    #     input_pkl="path/to/some_input_sequence.pkl",
    #     audio_dir="path/to/audio_directory",
    #     vecnorm_path=None,  # or "path/to/vecnormalize.pkl" if you use it
    # )


if __name__ == "__main__":
    main()
