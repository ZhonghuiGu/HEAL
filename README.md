# HEAL
Hierarchical Graph Transformer with Contrastive Learning for Protein Function Prediction \
The article is under submission

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

## Grad-CAM score mapped on protein structures
`python explainable_predictor.py --device $gpu_id
                     --task $task_name
                     --term_id $term_id
                     --pdb $pdb_file
                     --esm_main_path $esm_main_path
                     --esm1b_path $esm1b_model`
                     \
                     \
`$term_id` is the index of the GO-term function you concern. \
And the script will output a *.pdb file, and the b-factors of which are the grad-CAM contribution score. 
