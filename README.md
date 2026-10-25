# pokerchip instance segmentation project

## 以前のリポジトリについて
[前のリポジトリ](https://github.com/akisatoon1/yoloPokerchip)では
学習データ(メトリクスなど)やモデルの重みもgitで管理していてデータ量が多すぎたため, 
ソースコードのみを残してこちらのリポジトリに引き継いだ.

## 開発環境について
python仮想環境にて行う. 

### 作成
`python -m venv .venv`

### 起動, 終了
起動: `source .venv/bin/activate`

終了: `deactivate`

## 依存パッケージのインストール
仮想環境を起動してから行う.

`pip install -r requirements.txt`

## 環境変数の設定
```
export ROBOFLOW_API_KEY="example"
export ROBOFLOW_WORKSPACE="example"
export ROBOFLOW_PROJECT="example"
export WANDB_API_KEY="example"
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