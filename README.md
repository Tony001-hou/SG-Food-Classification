# SG-Food-Classification
This report shows classification of SG food using CNN

### Prerequisites

1. Create an environment
```
conda create --name sgfood python=3.8
conda activate sgfood
```
2. Install PyTorch=1.11.0 with this [link](https://pytorch.org/)
3. Install libraries
```
pip install -r requirements.txt
```

### Train and test

```
python main.py --model convnext_large
```


![CNN Result](https://github.com/Tony001-hou/SG-Food-Classification/blob/main/result.jpg)


