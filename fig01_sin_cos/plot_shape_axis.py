"""保存済みのsin/cosの曲線データから、グラフだけを作るプログラム。

【使う順番】neuromod_shape_axis.py で学習・評価 → このファイルで作図。
すでにshape_axis.npzがあれば、学習プログラムをもう一度実行する必要はない。
必要なのはNumPyとMatplotlib。PyTorchやGPU、学習済みモデルは読み込まない。
CSVだけには出力曲線がないので、作図にはNPZ（複数の配列を保存したファイル）を使う。

図は2行2列。上段が場所方式、下段が分布方式。
左列はsin/cosを学んだ出力、右列はノイズの混合率を変えた出力。
曲線は最初のシードのもの。凡例のMSEは全シード平均なので区別して読む。
混合係数λのグラフは表示しないが、NPZには保存されている。

PNG/PDFを保存するだけで、画面にウィンドウは出さない。
保存先は--outputで指定できる。すでにある図は上書きしない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
# --dataを省略したときは、この作図プログラムと同じ場所にあるNPZを読む。
DEFAULT_DATA = SCRIPT_DIR / "shape_axis.npz"
# --outputを省略したときの保存先。図2・図3と同じ、リポジトリ直下のfigures/。
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "figures" / "fig_shape_axis.png"
MODES = ("region", "shape")
SIN_COLOR = "#165E83"
COS_COLOR = "#D64527"


def load_results(path):
    """NPZを読み、必要な配列・設定・次元・欠損がないか検査する。"""
    # allow_pickle=False：モデル等のPythonオブジェクトは復元せず、数値と文字列だけ読む。
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"metadata_json", "modes", "seeds", "ws", "x", "target_sin",
                "target_cos"}
    for mode in MODES:
        required.update(f"{mode}_{name}" for name in
                        ("errors", "lambdas", "predictions"))
    missing = required - data.keys()
    # 一部しかない結果や、別形式のファイルから間違った図を描かないために確認する。
    if missing:
        raise ValueError(f"Missing result arrays: {', '.join(sorted(missing))}")
    metadata = json.loads(data["metadata_json"].item())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported result format version")
    if data["modes"].tolist() != list(MODES):
        raise ValueError("Expected region and shape results, in that order")
    for key in ("x", "ws", "seeds"):
        if data[key].ndim != 1 or data[key].size == 0:
            raise ValueError(f"{key} must be a nonempty one-dimensional array")
    n, nw, ns = len(data["x"]), len(data["ws"]), len(data["seeds"])
    # n=入力点数、nw=混合率の数、ns=シード数。
    # errorsは[シード, sin/cos]、lambdasは[シード, 混合率]、
    # predictionsは[シード, 混合率, 入力点]の形であることを確認する。
    shapes = {"target_sin": (n,), "target_cos": (n,)}
    for mode in MODES:
        shapes.update({f"{mode}_errors": (ns, 2),
                       f"{mode}_lambdas": (ns, nw),
                       f"{mode}_predictions": (ns, nw, n)})
    for key, shape in shapes.items():
        # NaNや無限大を曲線として描かない。誤った配列の並びにも気づけるようにする。
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} must be finite with shape {shape}")
    if not np.isfinite(data["x"]).all() or not np.isfinite(data["ws"]).all():
        raise ValueError("x and ws must be finite")
    if not all(w in data["ws"] for w in (0.0, 1.0)):
        raise ValueError("Results must include both w=0 and w=1")
    return data


def make_figure(data):
    """保存配列だけから2行2列の図を組み立てる。推論・学習は行わない。"""
    # 描画ライブラリはここで読み込む。ヘルプやデータ検査だけなら不要。
    import matplotlib
    matplotlib.use("Agg", force=True)  # 画面表示せずファイルへ描く方式。
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    xn, ws = data["x"], data["ws"]
    blend_colors = LinearSegmentedColormap.from_list(
        "indigo_vermilion", [SIN_COLOR, "#777777", COS_COLOR], N=257
    )
    for row, mode in enumerate(MODES):
        errs = data[f"{mode}_errors"]
        # 上段region・下段shape。曲線は先頭シード（通常7）のもの。
        # 曲線を3シードで平均しているわけではない。MSEの表示値は3シード平均。
        preds = dict(zip(ws, data[f"{mode}_predictions"][0]))
        ax = axes[row, 0]
        # 左列：破線は正解、色つき実線はノイズ条件を切り替えたモデルの出力。
        # ここで計算するsin/cosは正解の破線用で、ネットワークの再評価ではない。
        ax.plot(xn, np.sin(xn), "k--", lw=1, label="Targets")
        ax.plot(xn, np.cos(xn), "k--", lw=1)
        ax.plot(xn, preds[0.0], color=SIN_COLOR, lw=2,
                label=f"sin (MSE {errs[:, 0].mean():.3f})")
        ax.plot(xn, preds[1.0], color=COS_COLOR, lw=2,
                label=f"cos (MSE {errs[:, 1].mean():.3f})")
        if row == 0:
            ax.set_title("Learned outputs")
        ax = axes[row, 1]
        # 右列：ノイズの混合率wを変えた出力。w=0がsin側、w=1がcos側。
        # 記録済みの混合率のうち0、0.5、1を表示する（欠けた曲線は作らない）。
        for wv in (0.0, 0.5, 1.0):
            if wv in preds:
                ax.plot(xn, preds[wv], color=blend_colors(wv), lw=1.8,
                        label=f"w={wv:g}")
        if row == 0:
            ax.set_title("Interpolation")
        for ax in axes[row]:
            ax.set_xticks([-3, 0, 3])
            ax.set_yticks([-1, 0, 1])
            ax.grid(alpha=0.3)
            # 凡例をグラフの下側へ置き、線や数値に重ならないようにする。
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                      ncols=3, fontsize=11, frameon=False,
                      handlelength=1.4, handletextpad=0.5, columnspacing=0.9)
    # 余白を自動調整する。配置や色を変えても学習結果の値自体は変わらない。
    fig.tight_layout()
    return fig


def main():
    # 引数省略時は上のDEFAULT_DATA/DEFAULT_OUTPUTを使用。別の結果も指定して描ける。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="Saved shape_axis.npz (default: beside this script).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="PNG/PDF destination; existing files are not overwritten.")
    args = p.parse_args()
    source = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        p.error(f"No saved data at {source}; run neuromod_shape_axis.py first")
    if output.suffix.lower() not in (".png", ".pdf"):
        p.error("--output must end in .png or .pdf")
    if output.exists():
        # 既存の図を守る。描き直すときは--outputで別名を指定する。
        p.error(f"Refusing to overwrite {output}; choose another --output")
    try:
        data = load_results(source)
    except (ValueError, KeyError, TypeError) as exc:
        p.error(f"Invalid result file: {exc}")
    fig = make_figure(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as f:
            # 拡張子でPNG/PDFを選ぶ。保存直前に他の処理が同名を作っても上書きしない。
            fig.savefig(f, format=output.suffix[1:].lower(), dpi=150,
                        bbox_inches="tight")
    finally:
        # 保存の成否にかかわらず図を閉じ、メモリを解放する。
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"saved {output}")
    print("Saved results only; no training or network evaluation executed.")


if __name__ == "__main__":
    # 直接起動したときだけ保存する。importだけでは作図しない。
    main()
