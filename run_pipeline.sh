#!/bin/bash

python pipeline.py --stream /mnt/disks/datastorage/videos/keyboard_cat.mp4 \
				   -ow /mnt/disks/datastorage/weights/optical_weights.pth.tar \
				   -sw /mnt/disks/datastorage/weights/spatial_weights.pth.tar \
				   -mw /mnt/disks/datastorage/weights/motion_weights.pth.tar
