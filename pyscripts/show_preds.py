"""
学習済みモデルで画像フォルダを推論し, 予測結果を可視化した画像を保存するスクリプト.

ultralytics 標準の描画は見づらいため, box なら bounding box 付き,
seg ならマスクのみ (box・ラベルなし) で描画する.
色分けはデフォルトでクラスごと (同じクラスは同じ色, 違うクラスは違う色).
推論時の画像サイズ・デバイスは環境変数 IMAGE_SIZE / DEVICE (shell/env.sh) の値を使う.

使い方 (リポジトリのルートで実行):
    bash shell/run_pyscript.sh pyscripts/show_preds.py <weights_path> <imgs_dir> <box|seg> \\
        [--name NAME] [--conf CONF] [--iou IOU] [--color-mode {class,instance}]

例:
    bash shell/run_pyscript.sh pyscripts/show_preds.py \\
        runs/segment/pretrained/weights/best.pt dataset/pokerchip-v3/valid/images seg \\
        --name show-valid --conf 0.3 --iou 0.5

引数:
    weights_path  学習済みモデルの重みファイル (.pt) のパス
    imgs_dir      推論する画像が入ったディレクトリ
    box_or_seg    描画方法. box (bounding box を描画) / seg (マスクを描画)
    --name        出力先ディレクトリの名前 (デフォルト tmp)
    --conf        信頼度の閾値 (デフォルト 0.25)
    --iou         NMS の IoU 閾値 (デフォルト 0.7)
    --color-mode  色分けの方法. class (同じクラスは同じ色) / instance (インスタンスごとに別の色)
                  (デフォルト class)

出力 (runs/<task>/<name>/ 以下. task は segment / detect. 同名が既にあれば name2, name3, ...):
    pred_imgs/    可視化した予測画像
    labels/       予測結果の txt (YOLO 形式 + conf)
"""

import argparse
from ultralytics import YOLO

from lib.utils import show_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path")
    parser.add_argument("imgs_dir")
    parser.add_argument("box_or_seg")
    parser.add_argument("--name", default="tmp")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--color-mode", choices=["class", "instance"], default="class")
    args = parser.parse_args()

    model = YOLO(args.weights_path)
    show_predictions(
        name=args.name,
        model=model,
        imgs_dir=args.imgs_dir,
        box_or_seg=args.box_or_seg,
        conf=args.conf,
        iou=args.iou,
        color_mode=args.color_mode,
    )


if __name__ == "__main__":
    main()
