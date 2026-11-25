"""図2：食物・脅威・巣と、エージェントの開始位置を描く。

座標と巣の半径は、行動実験で使う world.py から直接読む。
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_common" / "core" / "tmp"))
from neuromod import world


def plot_environment(out_dir):
    objects = world.make_scripted_objects()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.91)

    # 薄い円は巣の領域。食物や脅威のマーカーの大きさは見やすさのための表示設定。
    for center in objects["shelter"]:
        ax.add_patch(Circle(center, world.SHELTER_REGION_RADIUS, facecolor="tab:blue",
                            edgecolor="tab:blue", alpha=.18, linewidth=1.3, zorder=1))
    styles = [("food", "o", "tab:green", 900),
              ("threat", "^", "tab:red", 1100),
              ("shelter", "s", "tab:blue", 1100)]
    handles = []
    for key, marker, color, size in styles:
        points = objects[key]
        ax.scatter(points[:, 0], points[:, 1], s=size, marker=marker, color=color,
                   edgecolor="black", linewidth=1.1, zorder=3)
        handles.append(Line2D([], [], marker=marker, color="none", markerfacecolor=color,
                              markeredgecolor="black", markersize=11, label=key.title()))

    # 開始位置は最初の巣と重なる。青い四角が見えるよう、黒丸を小さく重ねる。
    ax.scatter(*world.SCRIPTED_START, s=450, color="black", edgecolor="white",
               linewidth=1.1, zorder=4)
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor="black",
                          markersize=9, label="Agent (start)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=4,
              frameon=False, fontsize=12, columnspacing=1., handletextpad=.35, borderaxespad=0)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           xlabel="$x$", ylabel="$y$")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(True)
    ax.grid(alpha=.16, zorder=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(out_dir / "fig_environment.png", dpi=300, bbox_inches="tight", pad_inches=.12)
    fig.savefig(out_dir / "fig_environment.pdf", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures", help="図の保存先")
    args = parser.parse_args()
    plot_environment(args.out_dir)
    print(args.out_dir.resolve() / "fig_environment.png")
    print(args.out_dir.resolve() / "fig_environment.pdf")
