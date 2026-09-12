from os.path import join

from roboflow import Roboflow
from ultralytics import YOLO

from . import env
from .count_chips import save_counts_from_result
from .show_pred import save_pred_imgs


def download_dataset(project_name, version_n):
    """Roboflow からデータを YOLOv{YOLO_VERSION} 形式でダウンロードする。"""
    rf = Roboflow(api_key=env.ROBOFLOW_API_KEY)
    project = rf.workspace(env.ROBOFLOW_WORKSPACE).project(project_name)
    version = project.version(version_n)
    location = join(env.DATASET_ROOT, f"{project_name}-v{version_n}")
    return version.download("yolo26", location=location)


def train_model(name, dataset_dir):
    """学習済み YOLO に追加学習を行い、結果を返す。"""
    model = YOLO(env.YOLO_MODEL)
    model.train(
        name=name,
        data=join(dataset_dir, "data.yaml"),
        epochs=env.EPOCHS,
        patience=env.PATIENCE,
        fraction=env.FRACTION,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        batch=env.BATCH,
        cache=env.CACHE,
        amp=True,
    )
    return model


def evaluate(name, model, dataset_dir, split="test"):
    """dataset の test(default) セットを使って学習済みモデルを評価する。"""
    results = model.val(
        name=name,
        data=join(dataset_dir, "data.yaml"),
        split=split,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        save_txt=True,
        save_conf=True,
    )

    # 現状のデータセットでは, splitがvalのときデータはvalid/に保存されるため
    split_dir = "valid" if split == "val" else split
    save_counts_from_result(results, dataset_dir, split_dir=split_dir)


def show_predictions(name, model, imgs_dir, conf=0.25, iou=0.7):
    """学習済みモデルの推論を行い、結果を保存する。

    デフォルトの推論結果の画像がわかりづらいため。
    """
    results = model.predict(
        name=name,
        source=imgs_dir,
        conf=conf,
        iou=iou,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        save_txt=True,
        save_conf=True,
    )
    save_pred_imgs(results)
