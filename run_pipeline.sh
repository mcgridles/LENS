#!/bin/bash

python pipeline.py --stream /mnt/disks/datastorage/videos/keyboard_cat.mp4 \
				   --optical_weights /mnt/disks/datastorage/weights/optical_weights.pth.tar \
				   --spatial_weights /mnt/disks/datastorage/weights/spatial_weights.pth.tar \
				   --motion_weights /mnt/disks/datastorage/weights/motion_weights.pth.tar
