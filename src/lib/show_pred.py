import logging
from os import makedirs, path

logging.basicConfig(
    filename="save_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",  # 日本語メッセージ対策（Python 3.9+）
)
logger = logging.getLogger(__name__)


def save_pred_imgs(results, box_or_seg):
    """学習済みモデルの推論を行った画像のインスタンスセグメンテーションを可視化して保存する。"""
    for result in results:
        filename = path.basename(result.path)
        saved_path = path.join(result.save_dir, "pred_imgs", filename)
        makedirs(path.dirname(saved_path), exist_ok=True)
        try:
            save_pred_img(result, saved_path, box_or_seg)
        except ValueError as e:
            logger.error(f"Invalid argument for box_or_seg: {e}")


def save_pred_img(result, save_path, box_or_seg):
    """bounding boxを描画せず, instanceごとに異なる色でmaskを描画した画像を保存する。"""
    if box_or_seg == "box":
        result.plot(save=True, filename=save_path, boxes=True)
    elif box_or_seg == "seg":
        result.plot(
            save=True,
            filename=save_path,
            color_mode="instance",
            boxes=False,
            labels=False,
        )
    else:
        raise ValueError(f"box_or_seg の引数が間違っています. box_or_seg={box_or_seg}")
