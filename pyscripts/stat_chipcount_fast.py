"""stat_chipcount.py の高速版。

元の実装は iou ごとに model.predict() を回すため推論が iou_grid の数(19回)だけ走る。
iou が影響するのは後段の NMS だけなので、
  1. iou=1.0 (= 何も抑制しない) で推論を1回だけ行い、NMS 前の候補ボックスを保持
  2. 各 iou で torchvision の batched_nms を掛け直し、各 conf で枚数を数える
とすることで、推論コストを 1/19 にする。結果は元の実装と一致する。
(end2end=True の NMS-free モデルでは ultralytics と同様に NMS を掛けない。iou は結果に効かない)

使い方 (リポジトリのルートで実行):
    .venv/bin/python pyscripts/stat_chipcount_fast.py \\
        --weights pastData/sizes/bigdata/training-metrics/weights/best.pt \\
        --data dataset/bigdata-chip-v2/test

    # env.sh と仮想環境を読み込んで実行する場合
    bash shell/run_pyscript.sh pyscripts/stat_chipcount_fast.py --weights <重み> --data <dir>

引数:
    --weights  学習済みモデルの重みファイル (.pt) のパス。
               Ultralytics Platform で学習した場合はモデルページから best.pt を
               ダウンロードして、そのパスを渡す。
    --data     評価に使うデータセットの split ディレクトリのパス。
               次の構成になっていること (Roboflow の YOLO 形式の test/ や valid/ そのもの):
                   <data>/
                   ├── images/   評価画像 (xxx.jpg など)
                   └── labels/   正解ラベル (xxx.txt, 画像と同じファイル名)
               正解枚数は labels/*.txt の行数で数える。
    --imgsz    推論時の画像サイズ (デフォルト 1280。学習時と合わせる)
    --device   cpu (デフォルト) / 0 (GPU)
    --project  出力先の名前 (デフォルト iou-conf-search-fast)

出力 (runs/segment/<project>/ 以下):
    heatmap.png              iou x conf ごとの完全一致率のヒートマップ
    table.csv                上記の数値
    csv/iou-<iou>/*.csv      各 (conf, iou) での画像ごとの正解枚数と予測枚数
"""

import argparse
from os import path
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from lib.count_chips import count_lines_from_txt, save_csv_from_counts
from stat_chipcount import PREFIX, exact_rate, save_heatmap

IOU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ~ 0.95
CONF_GRID = [0.01, 0.02, 0.03] + IOU_GRID


def collect_candidates(model, images_dir, imgsz, device):
    """NMS を実質無効化して推論し、画像ごとの (boxes, scores, classes, valid) を返す。

    valid: 旧実装と同じく txt に書き出される検出かどうか。ultralytics の save_txt は
    マスク輪郭が 3 点未満の検出を書かないため、旧実装ではその検出は枚数に入らない。
    マスクは検出ごとに独立に計算されるので、NMS で何が残るかには依存しない。
    """
    cands = {}
    for r in model.predict(
        source=images_dir,
        imgsz=imgsz,
        device=device,
        conf=min(CONF_GRID),
        iou=1.0,  # IoU > 1.0 は起こらないので何も抑制されない
        max_det=100000,  # 候補を打ち切らない
        retina_masks=False,
        stream=True,
        save=False,
        verbose=False,
    ):
        b = r.boxes
        if r.masks is not None:
            valid = torch.tensor([len(seg) >= 3 for seg in r.masks.xyn], dtype=torch.bool)
        else:  # detect モデルは txt にボックスが必ず書かれる
            valid = torch.ones(len(b), dtype=torch.bool)
        cands[Path(r.path).stem + ".txt"] = (
            b.xyxy.cpu().float(),
            b.conf.cpu().float(),
            b.cls.cpu(),
            valid,
        )
    return cands


def search(cands, correct_label_dir, out_dir, end2end=False):
    table = pd.DataFrame(index=IOU_GRID, columns=CONF_GRID, dtype=float)
    correct = {
        p.name: count_lines_from_txt(p) for p in Path(correct_label_dir).glob("*.txt")
    }
    for iou in IOU_GRID:
        # 画像ごとに NMS 後に残り、かつ txt に書かれる検出の conf を保持
        kept = {}
        for name, (boxes, scores, cls, valid) in cands.items():
            if end2end:  # NMS-free モデルは ultralytics も NMS を掛けない
                keep = torch.arange(len(scores))
            else:  # クラス別 NMS (ultralytics と同じ)。無効な検出も NMS には参加する
                keep = batched_nms(boxes, scores, cls, iou)
            kept[name] = scores[keep][valid[keep]]
        for conf in CONF_GRID:
            # 検出が1つも無い画像 (kept が空 / 推論結果に無い) の予測枚数は 0
            counts_data = [
                (
                    name,
                    true,
                    int((kept[name] >= conf).sum()) if name in kept else 0,
                )
                for name, true in correct.items()
            ]
            save_csv_from_counts(
                counts_data, f"{out_dir}/csv/iou-{iou}/counts_c{conf}_i{iou}.csv"
            )
            table.loc[iou, conf] = exact_rate(counts_data)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="images/ と labels/ を含むdir")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="iou-conf-search-fast")
    args = parser.parse_args()

    model = YOLO(args.weights)
    end2end = getattr(model.model, "end2end", False)
    if end2end:
        print("WARNING: end2end(NMS-free) モデルなので iou は結果に影響しません")

    out_dir = f"{PREFIX}/{args.project}"
    cands = collect_candidates(
        model, path.join(args.data, "images"), args.imgsz, args.device
    )
    table = search(cands, path.join(args.data, "labels"), out_dir, end2end)
    table.to_csv(f"{out_dir}/table.csv")
    save_heatmap(table, out_path=f"{out_dir}/heatmap.png")


if __name__ == "__main__":
    main()
