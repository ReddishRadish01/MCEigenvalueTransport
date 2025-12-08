import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Pattern for files like: Loop_0_fluxTally.txt, Loop_7_fluxTally.txt, ...
pattern = "Loop_*_fluxTally.txt"

for filename in glob.glob(pattern):
    print(f"Processing {filename}...")

    # -------- 1) Read lattice data from text file --------
    rows = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue

            nums = []
            for token in line.replace(",", " ").split():
                try:
                    nums.append(float(token))
                except ValueError:
                    # ignore non-numeric tokens
                    continue

            if nums:
                rows.append(nums)

    if not rows:
        print(f"  Skipping {filename}: no numeric data found.")
        continue

    min_len = min(len(r) for r in rows)
    flux = np.array([r[:min_len] for r in rows], dtype=float)

    # -------- 2) Extract loop index from filename --------
    base = os.path.basename(filename)
    m = re.search(r"Loop_(\d+)_fluxTally", base)
    loop_idx = int(m.group(1)) if m else None

    # -------- 3) Build lattice indices --------
    Ny, Nx = flux.shape
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    dx = 0.8 * np.ones_like(Z)
    dy = 0.8 * np.ones_like(Z)
    dz = flux

    # -------- 4) Make 3D plot --------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(
        X.ravel(), Y.ravel(), Z.ravel(),
        dx.ravel(), dy.ravel(), dz.ravel(),
    )

    ax.set_xlabel("Lattice index (i)")
    ax.set_ylabel("Lattice index (j)")
    ax.set_zlabel("Flux tally")

    title = "3D Flux Tally Lattice"
    if loop_idx is not None:
        title += f" (Loop {loop_idx})"
    ax.set_title(title)

    plt.tight_layout()

    # -------- 5) Save PNG with same base name --------
    png_filename = os.path.splitext(filename)[0] + ".png"
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)  # important when looping many files

    print(f"  Saved {png_filename}")

print("Done.")
