# Low-light Environment Neural Surveillance (LENS)
Crime detection system for low-light environments. 

The systems uses modified versions of NVIDIA's (@NVIDIA) **FlowNet2** for calculating optical flow and Jeffrey Huang's (@jeffreyyihuang) **Two Stream Action Recognition** for performing the action recognition.

This project was done for a capstone for Northeastern University's class of 2020.

## Overview

## Installation
We recommend using Anaconda for managing the environment. This allows easy installation and keeps the environment separate from the rest of the system.

```bash
# Create conda environment and install dependencies
conda env create -f environemnt.yml
conda activate lens

# Install FlowNet2
cd flownet2-pytorch
bash install.sh
```

## Inference
Inference can be performed on an individual video or a video stream using OpenCV. 

```bash
python pipeline.py --stream /path/to/video.mov \
                   --model FlowNet2CSS \
                   --svm /path/to/svm/model.pkl \
                   --nb_classes 4 \
                   --skip_frames 1 \
                   --save /path/to/directory/ \
                   --optical_weights /path/to/FlowNet2-CSS_weights.pth.tar \
                   --spatial_weights /path/to/spatial_weights.pth.tar \
                   --motion_weights /path/to/motion_weights.pth.tar
```

If running on a video stream, the camera number should be passed to the `--stream` flag instead of a path to a file. Also, make sure the optical flow model architecture matches between the `--model` and `--optical_weights` flags.
