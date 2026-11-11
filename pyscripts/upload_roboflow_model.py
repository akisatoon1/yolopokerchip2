"""
学習済みモデルの重み (.pt) を Roboflow にアップロードするスクリプト.

ワークスペースは環境変数 ROBOFLOW_WORKSPACE のものが使われる.
--version を指定するとそのデータセットバージョンに紐づけてアップロードし,
指定しないとワークスペースレベル (versionless) で --model-name の名前でアップロードする.

参考: https://docs.roboflow.com/models/model-weights/upload-custom-weights

使い方 (リポジトリのルートで実行):
    bash shell/run_pyscript.sh pyscripts/upload_roboflow_model.py <weights_path> <project_name> \\
        [--version VERSION] [--model-name MODEL_NAME] [--model-type MODEL_TYPE]

例:
    # データセットバージョンに紐づけてアップロード
    bash shell/run_pyscript.sh pyscripts/upload_roboflow_model.py \\
        runs/segment/pretrained/weights/best.pt pokerchip --version 3

    # バージョンに紐づけずにアップロード
    bash shell/run_pyscript.sh pyscripts/upload_roboflow_model.py \\
        runs/segment/pretrained/weights/best.pt pokerchip --model-name pokerchip-seg-v1

引数:
    weights_path  アップロードする重みファイル (.pt) のパス
    project_name  アップロード先の Roboflow のプロジェクト名
    --version     紐づけるデータセットのバージョン番号 (省略時は versionless)
    --model-name  versionless でアップロードするときのモデル名 (--version 省略時は必須.
                  英字を1文字以上含み, 数字・ハイフンも使える)
    --model-type  Roboflow のモデルタイプ (デフォルト yolo26).
                  yolo26n-seg のように明示もできる. サイズ・タスクを省略すると重みから推定される
"""

import argparse
from os.path import basename, dirname

from roboflow import Roboflow

from lib import env


def upload_model(
    weights_path, project_name, version_n=None, model_type="yolo26", model_name=None
):
    """学習済みモデルの重み (.pt) を Roboflow にアップロードする。

    version_n を指定するとそのデータセットバージョンに紐づけてアップロードし,
    指定しないとワークスペースレベル (versionless) で model_name という名前でアップロードする。
    model_type のサイズ (n/s/m/...) とタスク (-seg など) を省略すると,
    roboflow SDK が重みから推定して補完する。
    """
    # SDK は model_path (ディレクトリ) と filename (model_path からの相対パス) を受け取る
    model_path = dirname(weights_path) or "."
    filename = basename(weights_path)

    rf = Roboflow(api_key=env.ROBOFLOW_API_KEY)
    workspace = rf.workspace(env.ROBOFLOW_WORKSPACE)
    if version_n is not None:
        version = workspace.project(project_name).version(version_n)
        version.deploy(model_type=model_type, model_path=model_path, filename=filename)
    else:
        workspace.deploy_model(
            model_type=model_type,
            model_path=model_path,
            project_ids=[project_name],
            model_name=model_name,
            filename=filename,
        )


def main():
    parser = argparse.ArgumentParser(
        description="学習済みモデルを Roboflow にアップロードする"
    )
    parser.add_argument(
        "weights_path", help="アップロードする重みファイル (.pt) のパス"
    )
    parser.add_argument(
        "project_name", help="アップロード先の Roboflow のプロジェクト名"
    )
    parser.add_argument(
        "--version", type=int, help="紐づけるデータセットのバージョン番号"
    )
    parser.add_argument(
        "--model-name", help="versionless でアップロードするときのモデル名"
    )
    parser.add_argument(
        "--model-type", default="yolo26", help="Roboflow のモデルタイプ"
    )
    args = parser.parse_args()

    if args.version is None and not args.model_name:
        parser.error("--version を指定しない場合は --model-name が必要です。")

    upload_model(
        weights_path=args.weights_path,
        project_name=args.project_name,
        version_n=args.version,
        model_type=args.model_type,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
