# Low-light Environment Neural Surveillance

Capstone project for Northeastern University engineering Class of 2020.

## Installation
```bash
# Create conda environment and install dependencies
conda env create -f environemnt.yml
conda activate lens

# Install FlowNet2
cd flownet2-pytorch
bash install.sh
```

## Inference
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
