#!/bin/bash
set -e
cd /home/jovyan/work
for seed in 15101 15102 15103; do
  python cnn15.py 96 "cnn15s${seed}" "$seed" 60 1.0 20 med 1.0
done
echo CNN15_DONE
