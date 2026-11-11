"""
Roboflow のデータセットをプロジェクト名とバージョンを指定してダウンロードするスクリプト.

ワークスペースは環境変数 ROBOFLOW_WORKSPACE のものが使われ,
DATASET_ROOT/<project_name>-v<version_n> に保存される.

使い方:
    shell/run_pyscript.sh pyscripts/download_roboflow_dataset.py <project_name> <version_n>

例:
    shell/run_pyscript.sh pyscripts/download_roboflow_dataset.py pokerchip 3
"""

import argparse
from lib import utils


def main():
    parser = argparse.ArgumentParser(description="Roboflow のデータセットをダウンロードする")
    parser.add_argument("project_name", help="Roboflow のプロジェクト名")
    parser.add_argument("version_n", type=int, help="データセットのバージョン番号")
    args = parser.parse_args()

    dataset = utils.download_dataset(
        project_name=args.project_name, version_n=args.version_n
    )
    print(f"ダウンロード先: {dataset.location}")


if __name__ == "__main__":
    main()
