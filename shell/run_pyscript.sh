#!/usr/bin/env bash

SCRIPT_DIR=$(cd $(dirname $0) ; pwd)

envsh="${SCRIPT_DIR}/env.sh"
if [[ -f "$envsh" ]]; then
    source "$envsh"
    echo "環境変数を設定するenv.sh を読み込みました。"
else
    echo "環境変数を設定するenv.sh が見つかりません。"
    exit 1
fi

actish="${SCRIPT_DIR}/activate.sh"
if [[ -f "$actish" ]]; then
    source "$actish"
    echo "仮想環境を有効にするactivate.sh を読み込みました。"
else
    echo "仮想環境を有効にするactivate.sh が見つかりません。"
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <python_script>"
    exit 1
else
    echo "実行するPythonスクリプト: $1"
    python "$1"
fi
