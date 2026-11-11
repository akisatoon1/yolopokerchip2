"""
事前学習用データセットで YOLO を学習し, 評価・予測の可視化を行い, 結果を wandb に記録するスクリプト.

処理の流れ:
    1. Roboflow から事前学習用 (ROBOFLOW_PRETRAINED_PROJECT) と
       自分のタスク用 (ROBOFLOW_TEST_PROJECT) のデータセットをダウンロード
    2. YOLO_MODEL を初期重みとして事前学習用データで学習 (name=pretrained)
    3. best.pt を評価
         - 事前学習用データの valid (name=val-pretrained. DEVICE=cuda のときのみ)
         - 自分のタスク用データの valid (name=val-mytask)
    4. 両データセットの valid/images を推論し, マスクを描画した画像を保存
       (name=show-val-predictions, show-mytask-predictions)
    5. 学習の経過と評価指標を wandb の project "yoloPokerchip" (run 名 pretrained) に記録

学習の設定 (YOLO_MODEL, EPOCHS, IMAGE_SIZE, DEVICE など) はすべて
shell/env.sh の環境変数で指定する. コマンドライン引数はない.
事前に `wandb login` しておくこと.

使い方 (リポジトリのルートで実行):
    bash shell/run_pyscript.sh pyscripts/training.py

出力 (runs/segment/ 以下):
    pretrained/               学習結果 (weights/best.pt など)
    val-pretrained/           事前学習用データでの評価結果 (counts.csv など)
    val-mytask/               自分のタスク用データでの評価結果 (counts.csv など)
    show-val-predictions/     事前学習用データの valid 画像の予測 (pred_imgs/, labels/)
    show-mytask-predictions/  自分のタスク用データの valid 画像の予測 (pred_imgs/, labels/)
"""

import os

import wandb
from ultralytics import YOLO, settings

from lib import env, utils

WANDB_PROJECT = "yoloPokerchip"


def log_eval_metrics(name, results):
    """評価結果の指標を wandb の summary に記録する。"""
    wandb.run.summary.update(
        {f"{name}/{k}": v for k, v in results.results_dict.items()}
    )


def main():
    # ultralytics 組み込みの wandb 連携を有効にする
    settings.update({"wandb": True})

    pretrained_dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_PRETRAINED_PROJECT,
        version_n=env.PRETRAINED_PROJECT_VERSION,
    )
    test_dataset = utils.download_dataset(
        project_name=env.ROBOFLOW_TEST_PROJECT,
        version_n=env.TEST_PROJECT_VERSION,
    )

    # 先に init しておくと ultralytics のコールバックはこの run に記録する
    run = wandb.init(
        project=WANDB_PROJECT,
        name="pretrained",
        config={
            "yolo_model": env.YOLO_MODEL,
            "pretrained_project_version": env.PRETRAINED_PROJECT_VERSION,
            "test_project_version": env.TEST_PROJECT_VERSION,
        },
    )
    # 学習終了時に ultralytics が run を finish する
    model = utils.train_model(
        name="pretrained",
        dataset_dir=pretrained_dataset.location,
    )
    model_best = YOLO(model.trainer.best)

    # 学習後の評価結果を同じ run に追記する
    wandb.init(project=WANDB_PROJECT, id=run.id, resume="must")
    if env.DEVICE == "cuda":
        # cpuのときはなぜかここでKilledされる
        results = utils.evaluate(
            name="val-pretrained",
            model=model_best,
            dataset_dir=pretrained_dataset.location,
            split="val",
        )
        log_eval_metrics("val-pretrained", results)
    results = utils.evaluate(
        name="val-mytask",
        model=model_best,
        dataset_dir=test_dataset.location,
        split="val",
    )
    log_eval_metrics("val-mytask", results)
    utils.show_predictions(
        name="show-val-predictions",
        model=model_best,
        imgs_dir=os.path.join(pretrained_dataset.location, "valid", "images"),
        box_or_seg="seg",
    )
    utils.show_predictions(
        name="show-mytask-predictions",
        model=model_best,
        imgs_dir=os.path.join(test_dataset.location, "valid", "images"),
        box_or_seg="seg",
    )
    wandb.finish()


if __name__ == "__main__":
    main()
