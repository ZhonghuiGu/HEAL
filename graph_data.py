import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import load_GO_annot
import numpy as np
import os
from utils import aa2idx
import sys
import esm

def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()

class GoTermDataset(Dataset):

    def __init__(self, set_type, task, AF2model=False):
        # task can be among ['bp','mf','cc']
        self.task = task
        if set_type != 'AF2test':
            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
        else:
            prot2annot, goterms, gonames, counts = load_GO_annot('data/nrSwiss-Model-GO_annot.tsv')
        goterms = goterms[self.task]
        gonames = gonames[self.task]
        output_dim = len(goterms)
        class_sizes = counts[self.task]
        mean_class_size = np.mean(class_sizes)
        pos_weights = mean_class_size / class_sizes
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
        # pos_weights = np.concatenate([pos_weights.reshape((len(pos_weights), 1)), pos_weights.reshape((len(pos_weights), 1))], axis=-1)
        # give weight for the 0/1 classification
        # pos_weights = {i: {0: pos_weights[i, 0], 1: pos_weights[i, 1]} for i in range(output_dim)}

        self.pos_weights = torch.tensor(pos_weights).float()


        self.processed_dir = 'data/processed'

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt")) 
        if set_type == 'AF2test':
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))["test_pdbch"]
        else:
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        self.y_true = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list])
        self.y_true = torch.tensor(self.y_true)

        if AF2model:
            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")
            
            graph_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_graph.pt"))
            self.graph_list += graph_list_af
            self.pdbch_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
            y_true_af = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list_af])
            
            self.y_true = np.concatenate([self.y_true, y_true_af],0)
            self.y_true = torch.tensor(self.y_true)


    def __getitem__(self, idx):

        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)
