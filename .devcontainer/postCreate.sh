#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -r requirements.txt
pip install black mdformat
sudo apt-get update
sudo apt-get install -y shfmt
