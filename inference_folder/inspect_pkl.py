import pickle
from pathlib import Path
import numpy as np

path = Path("model_v8.3.1.pkl")
with open(path, "rb") as f:
    data = pickle.load(f)

for k, v in data.items():
    arr = np.array(v)
    print(f"{k:15s} -> shape={arr.shape}, dtype={arr.dtype}, type={type(v)}")

# Optional: print a few elements from one key
print("\nFirst few values of 'smpl_poses':\n", data['smpl_poses'][:2])
