"""
bounding box (detect) 用データセットで YOLO を学習し, test 画像の予測を可視化するスクリプト.

処理の流れ:
    1. Roboflow から ROBOFLOW_BOX_PROJECT の BOX_PROJECT_VERSION をダウンロード
    2. YOLO_MODEL を初期重みとして学習 (name=training-box)
    3. best.pt で test/images を推論し, box を描画した画像を保存 (name=show-box)

学習の設定 (YOLO_MODEL, EPOCHS, IMAGE_SIZE, DEVICE など) はすべて
shell/env.sh の環境変数で指定する. コマンドライン引数はない.

使い方 (リポジトリのルートで実行):
    bash shell/run_pyscript.sh pyscripts/box_training.py

出力 (YOLO_MODEL が detect 用なら runs/detect/, seg 用なら runs/segment/ 以下):
    training-box/   学習結果 (weights/best.pt など)
    show-box/       test 画像の予測 (pred_imgs/, labels/)
"""

from os.path import join

from ultralytics import YOLO

from lib import env, utils


def main():
    dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_BOX_PROJECT,
        version_n=env.BOX_PROJECT_VERSION,
    )
    model = utils.train_model(
        name="training-box",
        dataset_dir=dataset.location,
    )
    model_best = YOLO(model.trainer.best)
    utils.show_predictions(
        name="show-box",
        model=model_best,
        imgs_dir=join(dataset.location, "test", "images"),
        box_or_seg="box",
    )


if __name__ == "__main__":
    main()
