# Dataset Distillation via the Wasserstein Metric

Official PyTorch implementation of the ICCV 2025 paper:
>[__"Dataset Distillation via the Wasserstein Metric"__](https://arxiv.org/abs/2311.18531)<br>
>Haoyang Liu, Yijiang Li, Tiancheng Xing, Peiran Wang, Vibhu Dalal, Luwei Li, Jingrui He, Haohan Wang<br>
>UIUC, UC San Diego, NUS

[`[Paper]`](https://arxiv.org/abs/2311.18531)  [`[Code]`](https://github.com/Liu-Hy/WMDD)  [`[Website]`](https://liu-hy.github.io/WMDD/) 

<div align=center>
<img width=100% src="./img/overview.png"/>
</div>

***Abstract.***
> Dataset Distillation (DD) aims to generate a compact synthetic dataset that enables models to achieve performance comparable to training on the full large dataset, significantly reducing computational costs. Drawing from optimal transport theory, we introduce WMDD (Dataset Distillation with Wasserstein Metric-based Feature Matching), a straightforward yet powerful method that employs the Wasserstein metric to enhance distribution matching.
>
> We compute the Wasserstein barycenter of features from a pretrained classifier to capture essential characteristics of the original data distribution. By optimizing synthetic data to align with this barycenter in feature space and leveraging per-class BatchNorm statistics to preserve intra-class variations, WMDD maintains the efficiency of distribution matching approaches while achieving state-of-the-art results across various high-resolution datasets. Our extensive experiments demonstrate WMDD's effectiveness and adaptability, highlighting its potential for advancing machine learning applications at scale.

## Run all

Modify the Pytorch source code according to this [train/README.md](train/README.md) \
Then, you can run the pretrain, recover, relabel, and eval stages with one script `run.sh`:
```bash
bash run.sh -x 1 -y 1 -d imagenette -u 0 -c 10 -r /home/user/data/ -n -w -b 10 -p
````
#### Tips:
- Prepare input datasets in the common ImageFolder format, and store them in the same parent folder specified by the `-r` flag
- `-x` is the experiment ID, which should be different for each run
- When you distill a particular dataset for the first time (with `-d` being imagenette, tiny-imagenet, or imagenet), 
  add the `-p` flag to pretrain a teacher model from scratch. The model will be saved in a path indexed by `-y`
- If you are satisfied with the teacher model, you can reuse it in subsequent runs by using the same `-y` value and *removing* the `-p` flag. This is recommended because pretraining can be time-consuming. 
  - E.g., if the first run gets good results, the second run could be `-x 2 -y 1`, to reuse teacher model `1` 
- You may need to tune `-b` (the regularization coefficient) for different datasets, e.g. 500 for tiny-imagenet
- This codebase currently doesn't support multi-GPU training, except for running the pretraining script separately. We will fix this later. 
- Set IPC with `-c`, and GPU index with `-u` (default to 0). Keep the `-n` and `-w` flags which are necessary to our method.

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

