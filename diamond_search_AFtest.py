import os
import gzip
from utils import load_GO_annot
import torch
import pickle as pkl
import numpy as np

prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
prot2annot1, _, _, _ = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")
#with open('data/nrPDB-GO_2019.06.18_sequences.fasta', 'r') as f:
with open('data/nrAF-Model-GO_sequences.fasta', 'r') as f:
    test_seq_text = f.read()

test_pdbch = torch.load('data/processed/AF2test_pdbch.pt')['test_pdbch']

test_seq_dict = {}
for idx,line in enumerate(test_seq_text.split('>')[1:]):
    pdbch = line.split(' nrSM')[0]
    if pdbch in test_pdbch:
        test_seq_dict[pdbch] = "".join(line.split(' nrSM')[1].split("\n"))

diamond_pred = []
for idx, pdbch in enumerate(test_pdbch):
    with open('tmp.fa','w') as f:
        f.write(f">tmp\n"+test_seq_dict[pdbch])
    print(idx, pdbch)
    os.system('diamond blastp -d data/nrAF_PDB_train_sequences.dmnd --more-sensitive -q tmp.fa --outfmt 6 qseqid sseqid bitscore > diamond.res')
    os.system('gzip diamond.res')

    mapping = {}
    #line = 'tmp     5E9C-A  781\n'
    with gzip.open('diamond.res.gz','rt') as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in mapping:
                mapping[it[0]] = {}
            mapping[it[0]][it[1]] = float(it[2])
    os.system('rm -f diamond.res.gz')
    os.system('rm -f diamond.res')
    allgos = np.zeros(2752)
    total_score = 0.0
    if mapping != {}:
        #print(mapping)
        for p_id, score in mapping['tmp'].items():
            if p_id in prot2annot.keys():
                allgos = allgos + np.concatenate([prot2annot[p_id]['mf'],prot2annot[p_id]['bp'],prot2annot[p_id]['cc']], -1) * score
            elif p_id in prot2annot1.keys():
                allgos = allgos + np.concatenate([prot2annot1[p_id]['mf'],prot2annot1[p_id]['bp'],prot2annot1[p_id]['cc']], -1) * score
            total_score += score
    #except:
    #    pass
    allgos = allgos / (total_score+0.01)
    print(np.where(allgos!=0.))
    diamond_pred.append(allgos)
diamond_pred = np.stack(diamond_pred)

with open('model/deepgoplusAF2_AF2test_predictions.pkl','rb') as f:
    deepgoplus_pred = pkl.load(f)

y_true = np.stack([np.concatenate([prot2annot1[pdb_c]['mf'],prot2annot1[pdb_c]['bp'],prot2annot1[pdb_c]['cc']],-1) for pdb_c in test_pdbch])
with open('model/deepgoplusAF2_diamond_AF2test_pred.pkl','wb') as f:
    pkl.dump([deepgoplus_pred, diamond_pred, y_true], f)


