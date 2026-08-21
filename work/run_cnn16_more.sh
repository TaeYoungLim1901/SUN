#!/bin/bash
set -e
cd /home/jovyan/work
for seed in 16102 16103; do
  python cnn16.py 96 cnn16s${seed} ${seed} 60 1.0 20 med 1.0
done
echo CNN16_DONE
