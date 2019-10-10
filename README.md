# Low-light Environment Neural Surveillance

Capstone project for Northeastern University engineering Class of 2020.

## Installation
```bash
# Create conda environment and install dependencies
conda env create --name lens python=3.7 --yes
conda env update -f environment.yml

conda activate lens

# Install FlowNet2
cd flownet2-pytorch
bash install.sh
