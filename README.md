# CDR_CIKM2023

Conference Paper (https://uobevents.eventsair.com/cikm2023//)

Title: CDR: Conservative Doubly Robust Learning for Debiased Recommendation

## Codes

Please refer to the python codes in the "CDR" folder for Conservative Doubly Robust experiments.

Please refer to the python codes in the "baseline" folder and baseline experiments.

## Datasets
Coat and Yahoo datasets are already included in this project.

KuaiRand dataset can be download at (https://kuairand.com/). You can preprocess the data using python script in data/kuairand.

## Expriments Setup

Our experimental environment is shown below:

sklearn version: 1.0.2

pytorch version: 1.9.0 + cu111

numpy version: 1.21.2

## Reference

We follow the previous study, which is shown below:

```
@inproceedings{li2023stabledr,
  title={StableDR: Stabilized Doubly Robust Learning for Recommendation on Data Missing Not at Random},
  author={Li, Haoxuan and Zheng, Chunyuan and Wu, Peng},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```


