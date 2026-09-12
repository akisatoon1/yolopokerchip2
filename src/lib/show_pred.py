import logging
from os import makedirs, path

import cv2

logging.basicConfig(
    filename="save_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",  # 日本語メッセージ対策（Python 3.9+）
)
logger = logging.getLogger(__name__)


def save_pred_imgs(results):
    """学習済みモデルの推論を行った画像のインスタンスセグメンテーションを可視化して保存する。"""
    for result in results:
        filename = path.basename(result.path)
        saved_path = path.join(result.save_dir, "pred_imgs", filename)
        makedirs(path.dirname(saved_path), exist_ok=True)
        try:
            save_pred_img(result, saved_path)
        except OSError as e:
            logger.error(f"Failed to save result: {e}")


def save_pred_img(result, save_path):
    """bounding boxを描画せず, instanceごとに異なる色でmaskを描画した画像を保存する。"""
    im_bgr = result.plot(color_mode="instance", boxes=False, labels=False)
    ok = cv2.imwrite(save_path, im_bgr)
    if not ok:
        raise OSError(f"保存に失敗しました: {save_path}")
