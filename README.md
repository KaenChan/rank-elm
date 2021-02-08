RankELM
===========================

## Citation
If you use the codes, please cite the following paper:
```
@inproceedings{chen2018flexible,
  title={Flexible ranking extreme learning machine based on matrix-centering transformation},
  author={Chen, Shizhao and Chen, Kai and Xu, Chuanfu and Lan, Long},
  booktitle={2018 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2018},
  organization={IEEE}
}
```

## Overview
RankELM is a ranking algorithm based on ELM and matrix-centering transformation. Query-level normalized loss function is used to avoid training a bias model. Matrix-centering transformation is used to optimize the loss function. The transformation greatly simplifies the learning process because of the symmetry and idempotence of centering matrix.

We implement three different kinds of ranking ELM algorithms based on the matrix-centering transformation:
- R-RankELM: regularized ranking ELM with better generalization performance;
- EI-RankELM: enhanced incremental ranking ELM which can incrementally add hidden nodes and can obtain a more compact network architecture;
- OS-RankELM: online sequential ranking ELM which can update the trained model using new training data.

## Demo

`run.m` give an example to train the ranking models. More datasets can be found in [LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor).

```
> run
Rank-ELM
rank_type    = pairwise
metric_type = NDCG
k_ndcg      = 0
N-hidden    = [400 600 800]
C           = -5

Fold1 N=400 C=2^-5     Time=(1.0300 0.0890) MAP=(0.7209 0.5271) NDCG=(0.7825 0.5586)	|	Best model N=400 C=2^-5 ValidMAP=0.5271 ValidNDCG=0.5586
Fold1 N=600 C=2^-5     Time=(0.0500 0.0250) MAP=(0.7329 0.5309) NDCG=(0.7859 0.5577)	|	Best model N=400 C=2^-5 ValidMAP=0.5271 ValidNDCG=0.5586
Fold1 N=800 C=2^-5     Time=(0.0570 0.0250) MAP=(0.7272 0.5515) NDCG=(0.7733 0.5914)	|	Best model N=800 C=2^-5 ValidMAP=0.5515 ValidNDCG=0.5914

The best model: N=800 C=2^-5 TrainTime= 0.0570; MAP=(0.7272 0.5515 0.4596) NDCG=(0.7733 0.5914 0.4848)

datasize    = [968 46]
elm_type    = rankelm
rank_type   = pairwise
metric_type = MAP
k_ndcg      = 5
N-hidden    = [400 600]

-----------------------------------------------------------------------------------------------
      | N    | C     | Train Time | MAP                    || Best | N     | C     | MAP      |
-----------------------------------------------------------------------------------------------
Fold1 | 400  | 2^-8  |  0.4730 s  | (0.7202 0.6131 0.4831) || Best | 400   | 2^-8  | 0.6131   |
Fold1 | 600  | 2^-15 |  0.4020 s  | (0.7228 0.5902 0.4541) || Best | 400   | 2^-8  | 0.6131   |
Best1 | 400  | 2^-8  |  0.4730 s  | (0.7202 0.6131 0.4831) ||

datasize     = [968 46]
elm_type     = i-rankelm
rank_type    = pairwise
metric_type  = MAP
k_ndcg       = 0
N-hidden     = 200
n_candidates = 1
Valid_interval = 10

1.52 s (1.43 s) | N: 10 / 200 | Train mse: 0.2339 | MAP 0.5806 - 0.5350 - 0.4731 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
1.90 s (0.32 s) | N: 20 / 200 | Train mse: 0.2228 | MAP 0.6416 - 0.4952 - 0.3921 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
2.26 s (0.32 s) | N: 30 / 200 | Train mse: 0.2144 | MAP 0.6330 - 0.4825 - 0.4381 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
2.58 s (0.28 s) | N: 40 / 200 | Train mse: 0.2110 | MAP 0.6740 - 0.4764 - 0.3931 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
2.90 s (0.28 s) | N: 50 / 200 | Train mse: 0.2050 | MAP 0.6724 - 0.4980 - 0.4138 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
3.22 s (0.29 s) | N: 60 / 200 | Train mse: 0.1989 | MAP 0.7196 - 0.4936 - 0.4407 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
3.54 s (0.29 s) | N: 70 / 200 | Train mse: 0.1950 | MAP 0.6794 - 0.4990 - 0.4443 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
3.87 s (0.29 s) | N: 80 / 200 | Train mse: 0.1903 | MAP 0.6947 - 0.5013 - 0.4898 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
4.23 s (0.32 s) | N: 90 / 200 | Train mse: 0.1881 | MAP 0.7452 - 0.5097 - 0.4840 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
4.59 s (0.32 s) | N: 100 / 200 | Train mse: 0.1831 | MAP 0.6918 - 0.5114 - 0.4655 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
4.96 s (0.33 s) | N: 110 / 200 | Train mse: 0.1805 | MAP 0.7105 - 0.5079 - 0.4459 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
5.33 s (0.32 s) | N: 120 / 200 | Train mse: 0.1785 | MAP 0.7200 - 0.5314 - 0.4825 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
5.71 s (0.33 s) | N: 130 / 200 | Train mse: 0.1764 | MAP 0.7663 - 0.5198 - 0.4674 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
6.03 s (0.28 s) | N: 140 / 200 | Train mse: 0.1756 | MAP 0.7660 - 0.5025 - 0.4696 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
6.35 s (0.28 s) | N: 150 / 200 | Train mse: 0.1743 | MAP 0.7768 - 0.5044 - 0.4718 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
6.67 s (0.28 s) | N: 160 / 200 | Train mse: 0.1730 | MAP 0.7725 - 0.5210 - 0.4666 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
6.98 s (0.28 s) | N: 170 / 200 | Train mse: 0.1717 | MAP 0.7771 - 0.5285 - 0.4936 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
7.31 s (0.29 s) | N: 180 / 200 | Train mse: 0.1704 | MAP 0.7721 - 0.5249 - 0.4596 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
7.65 s (0.30 s) | N: 190 / 200 | Train mse: 0.1680 | MAP 0.7720 - 0.5197 - 0.4538 || Valid N:10  MAP 0.5806 0.5350 0.4731 |
8.03 s (0.34 s) | N: 200 / 200 | Train mse: 0.1668 | MAP 0.7317 - 0.5363 - 0.4497 || Valid N:200  MAP 0.7317 0.5363 0.4497 |

flod1 best | n=200  | traintime=8.0450 s | MAP (0.7317 0.5363 0.4497) ||

Online Sequential Rank elm
metric_type = MAP
k_ndcg      = 5
N-hidden    = 100
C           = -7
Blocks      = 2

Fold0 N=100 C=2^-7 B=2  Time=(0.2950 0.0180) MAP=(0.7204 0.4843) NDCG=(0.7157 0.5925) | Best Valid N=100 C=2^-7 B=2  MAP=0.4843 NDCG=0.5925

Fold0 The best model: N=100 C=10^-7 B=2  TrainTime=0.295000; MAP=(0.7204 0.4843 0.4577) NDCG=(0.7157 0.5925 0.4047)

Online Sequential Rank ELM eig
metric_type = MAP
k_ndcg      = 5
N-hidden    = 200
Block       = 2
Train size  = [279 46]
Valid size  = [399 46]
Test size   = [290 46]

------------------------------------------------------------------------------------------------------------------------------------
      | N    | C     | Train Time | MAP                    | NDCG                   || Best | N     | C     | ValidMAP | ValidNDCG |
------------------------------------------------------------------------------------------------------------------------------------
Fold0 | 200  | 2^9   |  0.5270 s  | (0.9885 0.5559 0.4551) | (1.0000 0.5702 0.4834) || Best | 200   | 2^9   | 0.5559   | 0.5702    |
Best0 | 200  | 2^9   |  0.5270 s  | (0.9885 0.5559 0.4551) | (1.0000 0.5702 0.4834) ||
```

