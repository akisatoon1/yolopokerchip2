# pokerchip instance segmentation project

## 以前のリポジトリについて
[前のリポジトリ](https://github.com/akisatoon1/yoloPokerchip)では
学習データ(メトリクスなど)やモデルの重みもgitで管理していてデータ量が多すぎたため, 
ソースコードのみを残してこちらのリポジトリに引き継いだ.

## 開発環境について
python仮想環境を使う. 

`shell/setup_new_runpod.sh` を実行せよ.

## 環境変数の設定
`shell/env.sh` にて環境変数の設定をすること.

``` env.sh
# env.sh の例

#! /usr/bin/env bash
export API_KEY="example"
```

## ssh接続が切れてもプログラムを回し続ける
```
# Install tmux
apt-get update && apt-get install -y tmux

# Start a new TMUX session
tmux new -s model_training

# Navigate to your project directory
cd /workspace/my_project

# Start your training script
python train.py --epochs 100 --batch-size 32

# Detach from the session with Ctrl+B, then D
# You can now safely disconnect from the Pod

# Later, reconnect to the Pod and reattach
tmux attach -t model_training
```