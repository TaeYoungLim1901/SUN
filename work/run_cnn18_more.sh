#!/bin/bash
set -e
cd /home/jovyan/work
for seed in 18102 18103; do
  python cnn18.py 96 cnn18s${seed} ${seed} 60 1.0 20 med 1.0
done
echo CNN18_DONE
