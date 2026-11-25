"""sin/cosを、学習後の重みを変えずにノイズ条件で切り替える説明用実験。

【役割】学習と評価を行い、曲線を含むNPZと集計CSVを保存する。
図は同じフォルダの plot_shape_axis.py で描く。このファイルだけでは図は出ない。
入力・出力はともに1次元。

region（場所方式）：正規分布のノイズを与えるユニットの範囲を変える。
  範囲Aでsin、範囲Bでcosを学習。各範囲40個で、共通部分は16個。
  評価では各ユニットの標準偏差を sigma(w)=(1-w)A+wB と混合する。
shape（分布方式）：全ユニットで、一様分布ならsin、正規分布ならcosを学習。
  両分布の平均は0、分散は同じ。評価時は各ノイズ要素ごとに、
  確率wで正規分布、確率1-wで一様分布を選ぶ。

方式ごと・乱数シードごとに別のモデルを作り、各モデルがsinとcosの両方を学ぶ。
混合条件は学習せず、学習後に出力がどう変わるかを調べる。
出力が単純な線形混合になると決めつけず、保存曲線と推定係数から確認する。

必要な部品はNumPy・PyTorchと、同梱の_common/core/nnn/。
--out-dir を省略すると、このPythonファイルと同じ場所へ保存する。
既存結果は上書きせず停止する。モデルの重みそのものは保存しない。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
# クロッシングの処理を、同梱の_common/core/nnn/から読む。端末の作業場所には依存しない。
sys.path.insert(0, str(SCRIPT_DIR.parent / "_common" / "core"))

from nnn import activation

# ハイパーパラメータ＝学習前に決める設定。重みのように学習で求める値ではない。
HID = 64                   # 各隠れ層のユニット数。構成は1→64→64→1。
T_SAMPLES = 64             # 1回の推論で用いるノイズのサンプル数。
H_CROSS = 0.2              # クロッシングで用いる±hの閾値。
SIGMA0 = 0.8               # 基準ノイズの標準偏差。分散はこの値の二乗。
SEEDS = [7, 11, 23]        # 乱数の初期条件を変え、3回独立に学習する。
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SQRT3 = math.sqrt(3.0)     # 一様分布[-√3,√3]の分散を1にする係数。
EVAL_FORWARDS = 32         # 各混合率で32回推論し、曲線を平均する。
MODES = ("region", "shape")  # 場所方式・分布方式の両方を実行する。


class AxisNNN(nn.Module):
    """ノイズ条件condを変え、同じモデルでsinとcosを出力するNNN。

    cond=("region", 標準偏差ベクトル)：ユニット別の強さを指定する。
    cond=("shape", w)：正規分布を選ぶ確率を指定する。
    """

    def __init__(self):
        super().__init__()
        # Linearは「入力の重み付き和＋バイアス」。最後の層だけバイアスなし。
        self.fc1 = nn.Linear(1, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.fc3 = nn.Linear(HID, 1, bias=False)

    def _noise(self, shape, cond, device):
        """指定した配列の形状・装置（CPU/GPU）でノイズを生成する。"""
        kind, arg = cond
        if kind == "region":
            # 正規乱数にユニット別の標準偏差を掛ける。arg=0なら直接加えるノイズは0。
            return torch.randn(shape, device=device) * arg
        w = float(arg)
        gauss = torch.randn(shape, device=device)
        unif = (torch.rand(shape, device=device) * 2.0 - 1.0) * SQRT3
        # pickは0か1。分布の混合は値の加重平均ではなく、要素ごとに片方を選ぶ方法。
        pick = (torch.rand(shape, device=device) < w).float()
        return SIGMA0 * (pick * gauss + (1.0 - pick) * unif)

    def forward(self, x, cond):
        """推論1回分。各入力点について64サンプルを計算して平均する。"""
        dev = x.device
        # 第1層の入力を複製し、各サンプルへ独立したノイズを加える。
        a1 = self.fc1(x).unsqueeze(1).repeat(1, T_SAMPLES, 1)
        z1 = activation.CrossingSample.apply(
            a1 + self._noise(a1.shape, cond, dev), H_CROSS)
        a2 = self.fc2(z1)
        # ±hそれぞれで隣接サンプルの閾値判定を比較するのがクロッシング。
        # 第2層にも新しいノイズを加える。近似勾配の処理は共通ライブラリ側にある。
        z2 = activation.CrossingSample.apply(
            a2 + self._noise(a2.shape, cond, dev), H_CROSS)
        return self.fc3(z2).mean(dim=1)  # サンプル方向の平均。入力xごとに関数値1個を返す。


def region_fields(device):
    """場所方式の範囲A・B。ユニットの番号であり、環境内の座標ではない。"""
    a = torch.zeros(HID, device=device)
    b = torch.zeros(HID, device=device)
    a[:40] = SIGMA0        # A：番号0〜39の40個にノイズを与える。
    b[24:] = SIGMA0        # B：番号24〜63の40個。24〜39の16個を共有する。
    return a, b


def implied_lambda(y, ta, tb):
    """出力yが(1-λ)sin+λcosに最も近くなるλを、全入力点で最小二乗計算する。

    flattenは配列を1列に並べる操作。λは0〜1に切り詰めない。
    ノイズ条件の混合率wとは別の値で、λ=wになることを強制していない。
    """
    d = (tb - ta).flatten()
    return float(((y - ta).flatten() @ d) / (d @ d))


def run_mode(mode, seed, x, ys, yc, device, epochs, lr):
    """1方式・1シードの学習と評価。学習するのは混合前の2条件だけ。"""
    # シード＝乱数の初期条件。同じシード番号でも、方式ごとに新しいモデルを作る。
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = AxisNNN().to(device)
    fa, fb = region_fields(device)
    cond_a = ("region", fa) if mode == "region" else ("shape", 0.0)
    cond_b = ("region", fb) if mode == "region" else ("shape", 1.0)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        # 全入力点についてsinとcosのMSEをそれぞれ求め、足した誤差で1回更新する。
        # MSE＝予測と正解の差の二乗平均。小さいほど正解に近い。
        loss = (((net(x, cond_a) - ys) ** 2).mean()
                + ((net(x, cond_b) - yc) ** 2).mean())
        opt.zero_grad()   # 前回の勾配を消す。勾配は「重みをどちらへ変えるか」の手がかり。
        loss.backward()   # 誤差から各重みの勾配を計算する。
        opt.step()        # Adamという方法で重み・バイアスを更新する。
    net.eval()
    with torch.no_grad():
        # ここから評価のみ。重みは更新しないが、ノイズは引き続き生成する。
        # 端点の誤差は各1回の推論で測定し、次の混合曲線は各32回の平均で求める。
        err_a = float(((net(x, cond_a) - ys) ** 2).mean())
        err_b = float(((net(x, cond_b) - yc) ** 2).mean())
        lams, preds = [], {}
        for w in WS:
            # regionは標準偏差を混合し、shapeは2分布を選ぶ確率を変える。
            cond = (("region", (1 - w) * fa + w * fb) if mode == "region"
                    else ("shape", w))
            y = torch.stack([net(x, cond) for _ in range(EVAL_FORWARDS)]).mean(0)
            lams.append(implied_lambda(y, ys, yc))
            # 計算済みの曲線を保存用に取り出す。保存のための追加推論は行わない。
            preds[w] = y.cpu().numpy().ravel()
    return err_a, err_b, lams, preds, net


def main():
    # --epochs等は起動時に指定する引数。省略するとdefaultの設定を使う。
    p = argparse.ArgumentParser(
        description="Train/evaluate sin-cos NNNs and save curves without plotting.")
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR,
                   help="Result directory (default: this script's directory).")
    args = p.parse_args()
    if args.epochs < 1 or args.points < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("--epochs must be >= 1, --points >= 2, and --lr positive and finite")

    out = args.out_dir.expanduser().resolve()
    # 保存済みの結果があれば学習前に停止する。古いNPZ・CSVを消したり上書きしない。
    data_path = out / "shape_axis.npz"
    csv_path = out / "shape_axis.csv"
    for path in (data_path, csv_path):
        if path.exists():
            p.error(f"Refusing to overwrite {path}; choose another --out-dir")
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 入力xは[-π,π]の等間隔な256点（既定）、正解はsin(x)とcos(x)。
    # GPUが使えなければCPUを選ぶ。
    x = torch.linspace(-math.pi, math.pi, args.points,
                       device=device).unsqueeze(1)
    ys, yc = torch.sin(x), torch.cos(x)

    metadata = {
        # 後で条件を確認できるよう、設定とライブラリの版もNPZへ保存する。
        "format_version": 1,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "points": args.points,
        "hidden_units": HID,
        "samples": T_SAMPLES,
        "crossing_threshold": H_CROSS,
        "noise_std": SIGMA0,
        "evaluation_forwards": EVAL_FORWARDS,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "representative_seed": SEEDS[0],
    }
    data = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "modes": np.asarray(MODES),
        "seeds": np.asarray(SEEDS),
        "ws": np.asarray(WS),
        "x": x.cpu().numpy().ravel(),
        "target_sin": ys.cpu().numpy().ravel(),
        "target_cos": yc.cpu().numpy().ravel(),
    }
    for mode in MODES:
        # 2方式×3シード＝計6モデル。既定では1モデル6,000回の重み更新。
        errs, lam_curves, predictions = [], [], []
        for seed in SEEDS:
            t0 = time.time()
            ea, eb, lams, preds, _ = run_mode(mode, seed, x, ys, yc, device,
                                              args.epochs, args.lr)
            errs.append((ea, eb))
            lam_curves.append(lams)
            predictions.append(np.stack([preds[w] for w in WS]))
            print(f"[{mode} seed {seed}] ({time.time()-t0:.0f}s) "
                  f"err_sin={ea:.4f} err_cos={eb:.4f} "
                  f"lambda: {' '.join(f'{v:.2f}' for v in lams)}")
        data[f"{mode}_errors"] = np.asarray(errs)
        data[f"{mode}_lambdas"] = np.asarray(lam_curves)
        data[f"{mode}_predictions"] = np.asarray(predictions)

    # predictionsの並びは[シード, 混合率, 入力点]。全曲線を残して再作図に使う。
    # 数値・文字列だけを保存し、モデルやpickleオブジェクトは含めない。
    # NPZがあれば再学習・再評価なしで作図できるが、学習済み重みは復元できない。
    with data_path.open("xb") as f:
        np.savez_compressed(f, **data)
    print(f"saved {data_path}")

    # CSVは集計値のみ。曲線そのものがないので作図にはNPZが必要。
    with csv_path.open("x") as f:
        f.write("# two addressing axes, sin/cos (sample level, variance-matched)\n")
        f.write("# columns: mode(0=region,1=shape),seed,err_sin,err_cos,"
                + ",".join(f"lam_w{w:g}" for w in WS) + "\n")
        for mi, mode in enumerate(MODES):
            errs = data[f"{mode}_errors"]
            lam_curves = data[f"{mode}_lambdas"]
            for si, seed in enumerate(SEEDS):
                f.write(f"{mi},{seed},{errs[si,0]:.5f},{errs[si,1]:.5f},"
                        + ",".join(f"{v:.4f}" for v in lam_curves[si]) + "\n")
    print(f"saved {csv_path}")
    print("No figure generated. Run plot_shape_axis.py with this NPZ to plot.")


if __name__ == "__main__":
    # 直接実行したときだけ学習を開始する。importしただけでは開始しない。
    main()
