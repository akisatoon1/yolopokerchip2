"""
学習済みの重みを自分のタスク用データセットでファインチューニングし, 評価と可視化を行うスクリプト.

処理の流れ (main() の freezes に書いた値ごとに繰り返す):
    1. Roboflow から ROBOFLOW_TEST_PROJECT の TEST_PROJECT_VERSION をダウンロード
    2. スクリプト内の w_path の重みから, 先頭 freeze 層を固定して学習
       (optimizer=AdamW, lr0=0.001, name=finetune-mytask-freeze-<freeze>)
    3. valid セットで評価 (name=val-finetuned-freeze-<freeze>)
    4. valid/images を推論し, マスクを描画した画像を保存 (name=show-finetuned-preds-<freeze>)

コマンドライン引数はない. 初期重みは fintune_and_evaluate() 内の w_path,
固定する層数は main() 内の freezes を書き換えて指定する.
EPOCHS, IMAGE_SIZE, DEVICE などは shell/env.sh の環境変数で指定する.

使い方 (リポジトリのルートで実行):
    bash shell/run_pyscript.sh pyscripts/finetuning.py

出力 (runs/segment/ 以下):
    finetune-mytask-freeze-<freeze>/       学習結果 (weights/best.pt など)
    val-finetuned-freeze-<freeze>/         評価結果 (counts.csv など)
    show-finetuned-preds-<freeze>/         valid 画像の予測 (pred_imgs/, labels/)
"""

from os import path

from ultralytics import YOLO

from lib import env, utils


def finetune(name, weights_path, yaml_path, freeze):
    model = YOLO(weights_path)
    model.train(
        data=yaml_path,
        epochs=env.EPOCHS,
        patience=env.PATIENCE,
        fraction=env.FRACTION,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        name=name,
        amp=True,
        batch=-1,
        cache=env.CACHE,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.001,
    )
    return model


def fintune_and_evaluate(freeze):
    test_dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_TEST_PROJECT,
        version_n=env.TEST_PROJECT_VERSION,
    )

    w_path = "pastData/sizes/training2-for-imgsz-1280/training-online-dataset-2/weights/best.pt"
    model = finetune(
        name=f"finetune-mytask-freeze-{freeze}",
        weights_path=w_path,
        yaml_path=path.join(test_dataset.location, "data.yaml"),
        freeze=freeze,
    )

    utils.evaluate(
        name=f"val-finetuned-freeze-{freeze}",
        model=model,
        dataset_dir=test_dataset.location,
        split="val",
    )
    utils.show_predictions(
        name=f"show-finetuned-preds-{freeze}",
        model=model,
        imgs_dir=path.join(test_dataset.location, "valid", "images"),
        box_or_seg="seg",
    )


def main():
    freezes = [23]
    for freeze in freezes:
        fintune_and_evaluate(freeze)


if __name__ == "__main__":
    main()
