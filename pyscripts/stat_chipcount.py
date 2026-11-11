from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

from lib.count_chips import get_counts_data_from_label, save_csv_from_counts

PREFIX = "runs/segment"


def exact_rate(counts_data):
    """
    counts_data: list of (filename, correct_count, pred_count)
    予測枚数が正解枚数と完全一致した画像の割合を返す。
    """
    if len(counts_data) == 0:
        return np.nan
    correct = sum(1 for (_, true, pred) in counts_data if pred == true)
    return correct / len(counts_data)


def predict(project, weights_path, evaluated_dir, imgsz):
    model = YOLO(weights_path)

    iou_grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    conf_grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    conf_grid = [0.01, 0.02, 0.03] + conf_grid  # 低域を補強

    table = pd.DataFrame(index=iou_grid, columns=conf_grid, dtype=float)

    for iou in iou_grid:
        name = f"iou-{iou}"
        print("iou:", float(iou))
        model.predict(
            project=project,
            name=name,
            source=path.join(evaluated_dir, "images"),
            exist_ok=True,
            imgsz=imgsz,
            device="cpu",
            save_txt=True,
            save_conf=True,
            conf=0.01,
            iou=float(iou),
        )
        for conf in conf_grid:
            result_dir = f"{PREFIX}/{project}/{name}"
            counts_data = get_counts_data_from_label(
                pred_label_dir=f"{result_dir}/labels",
                correct_label_dir=path.join(evaluated_dir, "labels"),
                conf=conf,
            )
            save_csv_from_counts(
                counts_data, f"{result_dir}/csv/counts_c{conf}_i{iou}.csv"
            )
            table.loc[iou, conf] = exact_rate(counts_data)  # 追加

    return table  # 追加


def save_heatmap(table, out_path):
    """正答率テーブル(行=iou, 列=conf)をヒートマップ画像として保存し、best を返す。"""
    plt.figure(
        figsize=(max(8, len(table.columns) * 0.5), max(6, len(table.index) * 0.5))
    )
    ax = sns.heatmap(
        table.astype(float),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "exact count accuracy"},
    )
    ax.set_xlabel("conf")
    ax.set_ylabel("iou")
    ax.set_title("Exact count accuracy over conf / iou")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    best_iou = table.max(axis=1).idxmax()
    best_conf = table.loc[best_iou].idxmax()
    best_rate = table.loc[best_iou, best_conf]
    print(f"BEST: iou={best_iou} conf={best_conf} accuracy={best_rate:.3f}")
    print(f"saved: {out_path}")
    return best_iou, best_conf, best_rate


def main():
    project = "iou-conf-search"
    weights_path = "pastData/sizes/training2-for-imgsz-1280/training-online-dataset-2/weights/best.pt"
    evaluated_dir = "dataset/chips-counter-color-anws5-1ujue/valid"
    imgsz = 1280
    table = predict(project, weights_path, evaluated_dir, imgsz)
    save_heatmap(table, out_path=f"{PREFIX}/{project}/heatmap.png")


if __name__ == "__main__":
    main()
