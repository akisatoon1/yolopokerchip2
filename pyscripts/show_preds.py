"""
モデルの予測結果を可視化して画像を保存するスクリプト.
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
    args = parser.parse_args()

    model = YOLO(args.weights_path)
    show_predictions(
        name=args.name,
        model=model,
        imgs_dir=args.imgs_dir,
        box_or_seg=args.box_or_seg,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
