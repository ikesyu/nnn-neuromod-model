"""図3：採餌・回避・帰巣の基準ノイズ場と、それらが重なる範囲を描く。"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "_common" / "core"
sys.path[:0] = [str(CORE), str(CORE / "tmp")]
from neuromod import fields, world


def plot_reference_fields(out_dir):
    # 144個のニューロンに与えるノイズの標準偏差を、実験と同じ関数で決める。
    side = 12
    base_std = .8
    reference = fields.build_fields(world.CATEGORIES, hidden_dim=side**2, base_std=base_std,
                                    sigma=.22, theta_cut=.15, radius=.28)
    panels = {key: reference[key].numpy().reshape(side, side) for key in world.CATEGORIES}
    # 右端の各マスは、標準偏差が正になるノイズ場がそこで何種類あるかを示す。
    overlap = sum((panel > 0).astype(int) for panel in panels.values())
    labels = {"food": "foraging", "threat": "avoidance", "shelter": "sheltering"}

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    for ax, key in zip(axes[:3], world.CATEGORIES):
        image = ax.imshow(panels[key], origin="lower", cmap="magma", vmin=0, vmax=base_std,
                          extent=[0, 1, 0, 1])
        ax.set_xlabel(labels[key], fontsize=17)
        ax.set_xticks([])
        ax.set_yticks([])
    scale = fig.colorbar(image, ax=axes[:3], fraction=.03, pad=.03)
    scale.set_ticks([0, .4, .8])
    scale.ax.tick_params(labelsize=14)
    scale.ax.set_title(r"$\sigma_k$", fontsize=16, pad=10)

    image = axes[3].imshow(overlap, origin="lower", cmap="viridis", vmin=0, vmax=3,
                           extent=[0, 1, 0, 1])
    axes[3].set_xlabel("overlap (support count)", fontsize=17)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    scale = fig.colorbar(image, ax=axes[3], fraction=.046, pad=.05)
    scale.set_ticks([0, 1, 2, 3])
    scale.ax.tick_params(labelsize=14)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_field_layout.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_field_layout.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures", help="図の保存先")
    args = parser.parse_args()
    plot_reference_fields(args.out_dir)
    print(args.out_dir.resolve() / "fig_field_layout.png")
    print(args.out_dir.resolve() / "fig_field_layout.pdf")
