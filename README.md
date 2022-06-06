# Region Impurity and Prediction Uncertainty (CVPR 2022 Oral Presentation)
by [Binhui Xie](https://binhuixie.github.io), Longhui Yuan, [Shuang Li](https://shuangli.xyz), [Chi Harold Liu](https://scholar.google.com/citations?user=3IgFTEkAAAAJ&hl=en) and [Xinjing Cheng](https://scholar.google.com/citations?user=8QbRVCsAAAAJ&hl=en)

**[[Arxiv](https://arxiv.org/abs/2111.12940)]**
**[[Paper](https://arxiv.org/pdf/2111.12940.pdf)]**

🥳 We are happy to announce that RIPU was accepted at **CVPR22 Oral Presentation**.

## Overview
We propose a simple region-based active learning approach for semantic segmentation under a domain shift, aiming to automatically query a small partition of image regions to be labeled while maximizing segmentation performance. 

Our algorithm, Region Impurity and Prediction Uncertainty (RIPU), introduces a new acquisition strategy characterizing the spatial adjacency of image regions along with the prediction confidence. 
We show that the proposed region-based selection strategy makes more efficient use of a limited budget than image-based or point-based counterparts. 

![image](resources/framework.png)

We show some qualitative examples from the Cityscapes validation set, 
![image](resources/visualization_results.png)

and also visualize the queried regions to annotate.
![image](resources/visualization_active.png)

For more information on DAFormer, please check our **[[Paper](https://arxiv.org/pdf/2111.12940.pdf)]**.

## Citation
If you find this project useful in your research, please consider citing:
```latex
@InProceedings{Xie_2022_CVPR,
    author    = {Xie, Binhui and Yuan, Longhui and Li, Shuang and Liu, Chi Harold and Cheng, Xinjing},
    title     = {Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8068-8078}
}
```


## Prerequisites

- Python 3.7
- Pytorch 1.7.1
- torchvision 0.8.2

**Step-by-step installation**

```bash
conda create --name ADASeg -y python=3.7
conda activate ADASeg

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# this installs required packages
pip install -r requirements.txt

```

## Data Preparation

- Download [The Cityscapes Dataset](https://www.cityscapes-dataset.com/), [The GTAV Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/), and [The SYNTHIA Dataset](https://synthia-dataset.net/)

**The data folder should be structured as follows:**

```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gtav/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
│   └──	synthia
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   └──	
```

**Symlink the required dataset**

```bash
ln -s /path_to_cityscapes_dataset datasets/cityscapes
ln -s /path_to_gtav_dataset datasets/gtav
ln -s /path_to_synthia_dataset datasets/synthia
```

**Generate the label static files for GTAV/SYNTHIA Datasets by running** 

```bash
python datasets/generate_gtav_label_info.py -d datasets/gtav -o datasets/gtav/
python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
```

## Train

**We provide the training scripts in `scripts/` using single GPU.**

```bash
# training for GTAV to Cityscapes
sh gtav_to_cityscapes.sh

# training for SYNTHIA to Cityscapes
sh synthia_to_cityscapes.sh
```


## Evaluate

```bash
python test.py -cfg configs/gtav/deeplabv3plus_r101_RA.yaml resume results/v3plus_gtav_ra_5.0_precent/model_iter040000.pth OUTPUT_DIR results/v3plus_gtav_ra_5.0_precent
```


## Acknowledgements
This project is based on the following open-source projects. We thank their authors for making the source code publically available.
- [FADA](https://github.com/JDAI-CV/FADA)
- [SDCA](https://github.com/BIT-DA/SDCA)


## Contact

If you have any problem about our code, feel free to contact

- [binhuixie@bit.edu.cn](mailto:binhuixie@bit.edu.cn)

or describe your problem in Issues.
