HEAL
====
Hierarchical Graph Transformer with Contrastive Learning for Protein Function Prediction
---

<img src="model/GraphACL-BIO.png">

## Setup Environment

Clone the current repo

    git clone https://github.com/ZhonghuiGu/HEAL.git
    conda env create -f requirements.yml

You also need to install the relative packages to run ESM-1b protein language model. \
Please see [facebookresearch/esm](https://github.com/facebookresearch/esm#getting-started-with-this-repo-) for details. \
And the ESM-1b model weight we use can be downloaded [here](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt).


## Protein function prediction

    python predictor.py --task bp 
                        --device 0 
                        --pdb 3GDT-A.pdb 
                        --esm1b_model ../esm-1b/checkpoints/esm1b_t33_650M_UR50S.pt

`$task` can be among the three GO-term task -- `[bp, mf, cc]`. \
`$pdb` is the path of the pdb file. \
`$esm1b_model` is the path of the ESM-1b model weight.

#### output
```txt
The protein may hold the following functions of BP:
Possibility: 1.0 ||| Functions: GO:0072528, pyrimidine-containing compound biosynthetic process
Possibility: 0.91 ||| Functions: GO:0006206, pyrimidine nucleobase metabolic process
Possibility: 1.0 ||| Functions: GO:0072527, pyrimidine-containing compound metabolic process
Possibility: 0.99 ||| Functions: GO:1901137, carbohydrate derivative biosynthetic process
Possibility: 1.0 ||| Functions: GO:0006753, nucleoside phosphate metabolic process
Possibility: 0.78 ||| Functions: GO:0009173, pyrimidine ribonucleoside monophosphate metabolic process
Possibility: 1.0 ||| Functions: GO:0055086, nucleobase-containing small molecule metabolic process
Possibility: 0.79 ||| Functions: GO:0009130, pyrimidine nucleoside monophosphate biosynthetic process
Possibility: 1.0 ||| Functions: GO:0006793, phosphorus metabolic process
Possibility: 1.0 ||| Functions: GO:0009117, nucleotide metabolic process
Possibility: 0.97 ||| Functions: GO:0009156, ribonucleoside monophosphate biosynthetic process
Possibility: 1.0 ||| Functions: GO:1901293, nucleoside phosphate biosynthetic process
Possibility: 1.0 ||| Functions: GO:0019438, aromatic compound biosynthetic process
Possibility: 0.98 ||| Functions: GO:0046390, ribose phosphate biosynthetic process
Possibility: 0.98 ||| Functions: GO:0009124, nucleoside monophosphate biosynthetic process
Possibility: 1.0 ||| Functions: GO:0034654, nucleobase-containing compound biosynthetic process
Possibility: 0.95 ||| Functions: GO:0046112, nucleobase biosynthetic process
Possibility: 0.94 ||| Functions: GO:0006207, 'de novo' pyrimidine nucleobase biosynthetic process
Possibility: 1.0 ||| Functions: GO:0006221, pyrimidine nucleotide biosynthetic process
Possibility: 1.0 ||| Functions: GO:0006796, phosphate-containing compound metabolic process
Possibility: 0.81 ||| Functions: GO:0009129, pyrimidine nucleoside monophosphate metabolic process
Possibility: 0.95 ||| Functions: GO:0019856, pyrimidine nucleobase biosynthetic process
Possibility: 0.52 ||| Functions: GO:0006222, UMP biosynthetic process
Possibility: 1.0 ||| Functions: GO:0006220, pyrimidine nucleotide metabolic process
Possibility: 0.76 ||| Functions: GO:0009174, pyrimidine ribonucleoside monophosphate biosynthetic process
Possibility: 1.0 ||| Functions: GO:0090407, organophosphate biosynthetic process
Possibility: 0.98 ||| Functions: GO:0009260, ribonucleotide biosynthetic process
Possibility: 0.98 ||| Functions: GO:0009259, ribonucleotide metabolic process
Possibility: 0.83 ||| Functions: GO:0009112, nucleobase metabolic process
Possibility: 1.0 ||| Functions: GO:0019637, organophosphate metabolic process
Possibility: 1.0 ||| Functions: GO:0009165, nucleotide biosynthetic process
Possibility: 0.88 ||| Functions: GO:0009220, pyrimidine ribonucleotide biosynthetic process
Possibility: 1.0 ||| Functions: GO:1901135, carbohydrate derivative metabolic process
Possibility: 1.0 ||| Functions: GO:0009123, nucleoside monophosphate metabolic process
Possibility: 0.99 ||| Functions: GO:0019693, ribose phosphate metabolic process
Possibility: 0.99 ||| Functions: GO:0009161, ribonucleoside monophosphate metabolic process
Possibility: 0.9 ||| Functions: GO:0009218, pyrimidine ribonucleotide metabolic process
Possibility: 0.5 ||| Functions: GO:0016310, phosphorylation
```

## Model training

    cd data

Our data set can be downloaded from [here](https://pkueducn-my.sharepoint.com/:u:/g/personal/2001111563_stu_pku_edu_cn/EVGSjfSK9hJFg2nBgwrEtNQBaIiUrFPQaugUptA-QwYdFQ?e=ZmPZp0).

    unzip processed.zip

The dataset related files will be under `data/processed`. 
Files with prefix of `AF2` belong to AFch dataset, others belong to PDBch dataset.
Files with suffix of `pdbch` record the PDBid or uniprot accession of each protein, and files with suffix of `graph` contain the graph we constructed for each protein.  

```txt
AF2test_graph.pt  AF2train_graph.pt  AF2val_graph.pt  test_graph.pt  train_graph.pt  val_graph.pt
AF2test_pdbch.pt  AF2train_pdbch.pt  AF2val_pdbch.pt  test_pdbch.pt  train_pdbch.pt  val_pdbch.pt
```

To train the model:

    python train.py --device 0
                    --task bp 
                    --batch_size 64 
                    --suffix CLaf
                    --contrast True
                    --AF2model True   
                    
`$task` can be among the three GO-term task -- `[bp, mf, cc]`. \
`$suffix` is the suffix of the model weight file that will be saved. \
`$contrast` is whether to use contrastive learning. \
`$AF2model` is whether to add AFch training set for training.

For whom want to build the new dataset, the `*graph.pt` file contain the list of protein graphs. \
Each graph is built by `Pytorch Geometric`, and each graph has three attributes. \
`graph.edge_index \n [2, protein_len]` is edge index of residue pairs whose Ca are within 10 angstroms.\
`graph.native_x` is the one-hot embedding for each residue type. \
`graph.x` is the ESM-1b language embedding for each sequences.
