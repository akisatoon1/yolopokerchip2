import logging
from os import makedirs, path

logging.basicConfig(
    filename="save_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",  # 日本語メッセージ対策（Python 3.9+）
)
logger = logging.getLogger(__name__)


# box_or_seg ごとの color_mode のデフォルト値 (color_mode=None のとき使う)
# box: クラスごとに色分け, seg: インスタンスごとに色分け
DEFAULT_COLOR_MODES = {"box": "class", "seg": "instance"}


def save_pred_imgs(results, box_or_seg, color_mode=None):
    """学習済みモデルの推論を行った画像の予測結果を可視化して保存する。

    color_mode は "class" (同じクラスは同じ色) / "instance" (インスタンスごとに別の色)。
    None のときは box_or_seg ごとのデフォルト (DEFAULT_COLOR_MODES) を使う。
    """
    for result in results:
        filename = path.basename(result.path)
        saved_path = path.join(result.save_dir, "pred_imgs", filename)
        makedirs(path.dirname(saved_path), exist_ok=True)
        try:
            save_pred_img(result, saved_path, box_or_seg, color_mode)
        except ValueError as e:
            logger.error(f"Invalid argument: {e}")


def save_pred_img(result, save_path, box_or_seg, color_mode=None):
    """予測結果を描画した画像を保存する。

    box: bounding box を描画する。
    seg: bounding box・ラベルを描画せず, mask のみを描画する。
    色分けは color_mode ("class" / "instance") に従う。None なら box_or_seg ごとのデフォルト。
    """
    if box_or_seg not in DEFAULT_COLOR_MODES:
        raise ValueError(f"box_or_seg の引数が間違っています. box_or_seg={box_or_seg}")
    if color_mode is None:
        color_mode = DEFAULT_COLOR_MODES[box_or_seg]
    if color_mode not in ("class", "instance"):
        raise ValueError(f"color_mode の引数が間違っています. color_mode={color_mode}")

    if box_or_seg == "box":
        result.plot(save=True, filename=save_path, boxes=True, color_mode=color_mode)
    else:
        result.plot(
            save=True,
            filename=save_path,
            color_mode=color_mode,
            boxes=False,
            labels=False,
        )
