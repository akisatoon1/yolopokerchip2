import csv
import os
from pathlib import Path


def count_lines_from_txt(path, conf=None):
    """txt の有効行数を数える。conf_thres 指定時は末尾(信頼度)>=thres の行のみ。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません。")
    n = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if conf is None:
                if (
                    len(p) >= 7
                ):  # class + 最低3点(アノテーション)の最低7つ要素が必要だから
                    n += 1
            else:
                if len(p) >= 8 and float(p[-1]) >= conf:  # 予測: 信頼度で足切り
                    n += 1
    return n


def save_csv_from_counts(counts_data, csv_path):
    """counts_data を csv に保存する。"""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_filename", "correct_count", "pred_count"])
        writer.writerows(counts_data)


def get_counts_data_from_label(pred_label_dir, correct_label_dir, conf=0.25):
    """pred_label_dir と correct_label_dir の各 txt ファイルの行数を数えて返す。"""
    counts_data = []
    for correct_path in Path(correct_label_dir).glob("*.txt"):
        filename = correct_path.name
        pred_path = Path(pred_label_dir) / filename

        correct_count = count_lines_from_txt(correct_path)
        pred_count = (
            count_lines_from_txt(pred_path, conf=conf) if pred_path.exists() else 0
        )

        counts_data.append((filename, correct_count, pred_count))
    return counts_data


def save_counts_from_result(val_results, dataset_dir, split_dir):
    pred_dir = Path(val_results.save_dir) / "labels"
    correct_dir = Path(dataset_dir) / split_dir / "labels"

    counts_data = get_counts_data_from_label(pred_dir, correct_dir)

    csv_path = Path(val_results.save_dir) / "counts.csv"
    save_csv_from_counts(counts_data, csv_path)
