"""
チップの色を区別するデータセットで YOLO を学習し, 評価と予測の可視化を行うスクリプト.

処理の流れ:
    1. Roboflow から ROBOFLOW_COLOR_PROJECT の COLOR_PROJECT_VERSION をダウンロード
    2. YOLO_MODEL を初期重みとして学習 (name=color-trained)
    3. best.pt を valid セットで評価 (name=val-color)
    4. valid/images を推論し, マスクを描画した画像を保存 (name=show-color-preds)

学習の設定 (YOLO_MODEL, EPOCHS, IMAGE_SIZE, DEVICE など) はすべて
shell/env.sh の環境変数で指定する. コマンドライン引数はない.

使い方 (リポジトリのルートで実行):
    bash shell/run_pyscript.sh pyscripts/color_train.py

出力 (runs/segment/ 以下):
    color-trained/     学習結果 (weights/best.pt など)
    val-color/         評価結果 (counts.csv: 画像ごとの正解枚数と予測枚数 など)
    show-color-preds/  valid 画像の予測 (pred_imgs/, labels/)
"""

from os.path import join

from ultralytics import YOLO

from lib import utils
from lib.env import ROBOFLOW_COLOR_PROJECT, COLOR_PROJECT_VERSION


def main():
    dataset = utils.download_dataset(
        project_name=ROBOFLOW_COLOR_PROJECT,
        version_n=COLOR_PROJECT_VERSION,
    )
    model = utils.train_model(
        name="color-trained",
        dataset_dir=dataset.location,
    )

    best_model = YOLO(model.trainer.best)

    utils.evaluate(
        name="val-color",
        model=best_model,
        dataset_dir=dataset.location,
        split="val",
    )
    utils.show_predictions(
        name="show-color-preds",
        model=best_model,
        imgs_dir=join(dataset.location, "valid", "images"),
        box_or_seg="seg",
    )


if __name__ == "__main__":
    main()
