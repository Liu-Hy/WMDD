# Dataset Distillation with Feature Matching through the Wasserstein Metric

Official PyTorch implementation of paper:
>[__"Dataset Distillation with Feature Matching through the Wasserstein Metric"__](https://arxiv.org/abs/2311.18531)<br>
>Haoyang Liu, Yijiang Li, Tiancheng Xing, Peiran Wang, Vibhu Dalal, Luwei Li, Jingrui He, Haohan Wang<br>
>UIUC, UC San Diego, Nanjing University

[`[Paper]`](https://arxiv.org/abs/2311.18531)  [`[Code]`](https://github.com/Liu-Hy/WMDD)  [`[Website]`](https://liu-hy.github.io/WMDD/) 

<div align=center>
<img width=100% src="./img/overview.png"/>
</div>

***Abstract.***
> Dataset Distillation (DD) aims to generate a compact synthetic dataset that enables models to achieve performance comparable to training on the full large dataset, significantly reducing computational costs. Drawing from optimal transport theory, we introduce WMDD (Dataset Distillation with Wasserstein Metric-based Feature Matching), a straightforward yet powerful method that employs the Wasserstein metric to enhance distribution matching.
>
> We compute the Wasserstein barycenter of features from a pretrained classifier to capture essential characteristics of the original data distribution. By optimizing synthetic data to align with this barycenter in feature space and leveraging per-class BatchNorm statistics to preserve intra-class variations, WMDD maintains the efficiency of distribution matching approaches while achieving state-of-the-art results across various high-resolution datasets. Our extensive experiments demonstrate WMDD's effectiveness and adaptability, highlighting its potential for advancing machine learning applications at scale.

## Run all

Modify the Pytorch source code according to this [train/README.md](train/README.md)
```bash
bash run.sh -x 2 -y 1 -d imagenette -u 0 -c 10 -r /home/user/data/ -n -w -b 3.0
````

> 📌 **Note**: This repo contains a historical version of WMDD that contains its core functionalities, enough to 
> reproduce results that are very close to our reported results. However, the authors are struggling with deadlines.
> We are testing the code for the current, improved version, 
> and will make it available with better documentation and code quality in around 4 weeks. 
> If you are interested in our work, please star ⭐ this repo to get notified for future updates!

## Citation

If you find our code useful for your research, please cite our paper.

```
@misc{liu2025wmdd,
      title={Dataset Distillation via the Wasserstein Metric}, 
      author={Haoyang Liu and Yijiang Li and Tiancheng Xing and Peiran Wang and Vibhu Dalal and Luwei Li and Jingrui He and Haohan Wang},
      year={2025},
      eprint={2311.18531},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.18531}, 
}
```

