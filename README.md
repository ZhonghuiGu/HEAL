HEAL
====
Hierarchical Graph Transformer with Contrastive Learning for Protein Function Prediction \
---

![image](https://github.com/ZhonghuiGu/HEAL/blob/main/model/GraphACL-BIO.png)

## Dependencies
`conda env create -f requirements.yml`

You also need to install the relative packages to run ESM-1b protein language model. \
See 


## Protein function prediction

`python predictor.py --task bp 
                     --device 0 
                     --pdb 3GDT-A.pdb 
                     --esm1b_model ../esm-1b/checkpoints/esm1b_t33_650M_UR50S.pt`
 \
 \
`$task_name` can be among `[bp, mf, cc]`. \
`$pdb` is the path of the pdb file. \
`$esm1b_model` is the path of ESM1b model.


## Model training
`cd data`
Our data set can be downloaded from https://pkueducn-my.sharepoint.com/:u:/g/personal/2001111563_stu_pku_edu_cn/EVGSjfSK9hJFg2nBgwrEtNQBaIiUrFPQaugUptA-QwYdFQ?e=ZmPZp0 
`python train.py --device 0
                 --task bp 
                 --batch_size 64 
                 --suffix CLaf
                 --contrast True
                 --AF2model True`
 
