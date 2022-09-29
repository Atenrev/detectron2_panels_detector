# Panels detector
## Description
This repository contains the associated files to train a comic panels segmenter using the Detectron2 library.

## Set up
- Create a conda environment with Python 3.9
```
conda create -n panels_detector python=3.9
conda activate panels_detector
```

- Install [Pytorch 1.9.1 cuda 10.2](https://pytorch.org/get-started/previous-versions/)
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
```

- Install Detectron2
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
```

- Install other dependencies and fix distutils problem
```
pip install opencv-python fiftyone setuptools==59.5.0
```

## References
- [Object Detection for Comics using Manga109 Annotations](https://arxiv.org/pdf/1803.08670.pdf)
- [Multi-task Model for Comic Book Image Analysis](https://link.springer.com/chapter/10.1007/978-3-030-05716-9_57)
- [C2VNet: A Deep Learning Framework Towards Comic Strip to Audio-Visual Scene Synthesis](https://link.springer.com/chapter/10.1007/978-3-030-86331-9_11)
- [Extraction of Frame Sequences in the Manga Context](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9327968)
