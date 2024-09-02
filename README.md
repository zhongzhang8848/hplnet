# PyTorch project about cross-domain person ReID
![](https://img.shields.io/badge/python->=v3.7-blue)![](https://img.shields.io/badge/pytorch->=v1.6-red)

## Requirements
Examples:
```
conda create -n 'env_name' python=3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.25.0
pip install scikit-learn
```

## Getting started
### Prepare Data
Download the person datasets [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) and [MSMT17 Dataset](https://arxiv.org/abs/1711.08565).
Then unzip them under the directory like:
```
datasets
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
    └── MSMT17_V1
```
Download the testing data, the link is:
```
链接：https://pan.baidu.com/s/1XPuK4wfFDGVJ3Jt2fH2d2w?pwd=zwjm 
提取码：zwjm 
```

### Test on MSMT17-to-Market1501
```
python test.py --data_dir data_path --fea_dir model_feature_path --output_dir output_path
```

## Contributing
Any kind of enhancement or contribution is welcomed.

