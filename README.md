# Robust Multimodal Survival Prediction with the Latent Differentiation Conditional Variational AutoEncoder
<details>
<summary>
  <b>Robust Multimodal Survival Prediction with the Latent Differentiation Conditional Variational AutoEncoder</b>, CVPR 2025.
  <!-- <a href="https://arxiv.org/abs/2306.08330" target="blank">[arxiv]</a>
  <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Multimodal_Optimal_Transport-based_Co-Attention_Transformer_with_Global_Structure_Consistency_for_ICCV_2023_paper.pdf" target="blank">[paper]</a> -->
  <br><em>Junjie Zhou, Jiao Tang, Yingli Zuo, Peng Wan, Daoqiang Zhang, WEI SHAO</em></br>
</summary>
<!-- 
```bash
@InProceedings{Xu_2023_ICCV,
    author    = {Xu, Yingxue and Chen, Hao},
    title     = {Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21241-21251}
}
``` -->
</details>

<img src="assets/Figure_ldvae.jpg" width="1500px" align="center" />

**Summary:** Here is the official implementation of the paper "Robust Multimodal Survival Prediction with the Latent Differentiation Conditional Variational AutoEncoder".

## Table of Contents
- [Introduction](#introduction)
- [Data preparation](#data-preparation)
- [Requirements](#requirements)
- [Usage](#Usage)
- [Acknowledgement](#acknowledgement)
<!-- - [License & Citation](#license--citation) -->

## Introduction

The integrative analysis of histopathological images and genomic data has received increasing attention for survival prediction of human cancers. However, the existing studies always hold the assumption that full modalities are available. As a matter of fact, the cost for collecting genomic data is high, which sometimes makes genomic data unavailable in testing samples. A common way of tackling such incompleteness is to generate the genomic representations from the pathology images. Nevertheless, such strategy still faces the following two challenges: (1) The gigapixel whole slide images (WSIs) are huge and thus hard for representation. (2) It is difficult to generate the genomic embeddings with diverse function categories in a unified generative framework. To address the above challenges, we propose a Conditional Latent Differentiation Variational AutoEncoder (LD-CVAE) for robust multimodal survival prediction, even with missing genomic data. Specifically, a Variational Information Bottleneck Transformer (VIB-Trans) module is proposed to learn compressed pathological representations from the gigapixel WSIs. To generate different functional genomic features, we develop a novel Latent Differentiation Variational AutoEncoder (LD-VAE) to learn the genomic and function-specific posteriors for the genomic embeddings with diverse functions. Finally, we use the product-of-experts technique to integrate the genomic posterior and image posterior for the joint latent distribution estimation in LD-CVAE. We test the effectiveness of our method on five different cancer datasets, and the experimental results demonstrate its superiority in both complete and missing modality scenarios. 


## Data preparation
### WSIs
1. Downloading the original WSI data from [TCGA](https://portal.gdc.cancer.gov/)
2. Preprocessing WSI data by [CLAM](https://github.com/mahmoodlab/CLAM).

The final structure of datasets should be as following:
```
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

### Genomics
1. The genomic data are provided in the `datasets_csv` folder. 

## Requirements

1. Create conda environment.
```
conda create -n robust python=3.9
conda activate robust
```
2. Install the required packages.
```
pip install -r requirements.txt
```
3. Check the installed packages.
```
conda list
```

## Usage
Experiments can be run using the following generic command-line:
```bash
CUDA_VISIBLE_DEVICES=<CUDA_IDX> python main.py --split_dir <SPLIT_DIR> --data_root_dir <DATA_DIR> --feature_extractor <FEATURE> --wsi_encoding_dim <WSI_DIM> --fusion <FU_TYPE> --mode <MODE> --model_type <MODE_TYPE> --g_model_type <G_MODE_TYPE> --g_condition --generator --warm_epoch <WARM_EPOCH>--max_epochs <MAX_EPOCHES>
```

## Acknowledgement
We would like to thank the following repositories for their great works:
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [CTransPath](https://github.com/Xiyue-Wang/TransPath)
- [MCAT](https://github.com/mahmoodlab/MCAT)

<!-- ## License & Citation
This project is licensed under the Apache-2.0 License.

If you find this work useful, please cite our paper: -->
