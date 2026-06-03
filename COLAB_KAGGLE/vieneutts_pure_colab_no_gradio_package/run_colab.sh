#!/usr/bin/env bash
set -e
apt-get update -y
apt-get install -y ffmpeg espeak-ng
pip install -U -r requirements.txt
python app.py
