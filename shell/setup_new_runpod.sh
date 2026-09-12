#!/usr/bin/env bash

python -m venv .venv --system-site-packages
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
