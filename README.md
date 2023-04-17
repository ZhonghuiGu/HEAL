# HEAL
Hierarchical Graph Transformer with Contrastive Learning for Protein Function Prediction \
The article is under submission

![image](https://github.com/ZhonghuiGu/HEAL/model/GraphACL-BIO.png)

## Dependencies
`conda env create -f requirements.yml`

You also need to install the relative packages to run ESM-1b protein language model. \
See 


## Protein function prediction

`python predictor.py --device $gpu_id
                     --task $task_name
                     --pdb $pdb_file
                     --esm_main_path $esm_main_path
                     --esm1b_path $esm1b_model`
 \
 \
`$task_name` can be among `[bp, mf, cc]`. \
`$pdb_file` is the path of the pdb file. \
`$esm_main_path` is the path of esm packeges. \
`$esm1b_model` is the path of ESM1b model.
 
