#!/bin/bash

python -W ignore pipeline.py --stream /mnt/disks/datastorage/videos/cold_day/demos/shooting_ballroom_light6.mov \
                             --nb_classes 3 \
                             --skip_frames 1 \
                             --buffer_size 100 \
                             --svm /home/mlp/two-stream-action-recognition/demos_svm.pkl \
                             --save /mnt/disks/datastorage/predictions/ \
                             --model FlowNet2CSS \
                             --optical_weights /mnt/disks/datastorage/weights/FlowNet2-CSS.pth.tar \
                             --spatial_weights /home/mlp/two-stream-action-recognition/record/spatial/model_best_DEMO.pth.tar \
                             --motion_weights /home/mlp/two-stream-action-recognition/record/motion/model_best_DEMO.pth.tar