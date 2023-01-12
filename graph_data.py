import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import pmap_multi, pmap_single, protein_graph, load_GO_annot
import numpy as np
import os
from utils import aa2idx
import sys
sys.path.append('../esm-1b/facebookresearch_esm_main/')  ## add the dir path of esm
import esm

def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()

class GoTermDataset(Dataset):

    def __init__(self, set_type, task, AF2model=False, SWmodel=False):
        # task can be among ['bp','mf','cc']
        self.task = task
        prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
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
        if not os.path.exists(self.processed_dir):
            self.model, alphabet = esm.pretrained.load_model_and_alphabet_local(
                "/home/lhlai_pkuhpc/lustre3/zhgu/esm-1b/checkpoints/esm1b_t33_650M_UR50S.pt") ## add the dir path of esm1b model
            self.batch_converter = alphabet.get_batch_converter()
            self.model = self.model.to(f"cuda:5")
            self.model.eval()
            self.process()

        if set_type == "train" or set_type == "val":
            self.out_path = os.path.join(self.processed_dir, "train_graph.pt")
            self.val_out_path = os.path.join(self.processed_dir, "val_graph.pt")
        
        elif set_type == "test":
            self.test_out_path = os.path.join(self.processed_dir, "test_graph.pt")
           
        self.AF_embedding = True
        #self.contact_cutoff = 0.5
        self.n_jobs = 10

        if not os.path.exists(os.path.join(self.processed_dir, "train_graph.pt")):
            self.process()

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt")) 
        self.y_true = [graph_dict[f'y_true_{task}'] for graph_dict in torch.load('data/processed/trainset.pt']
        self.y_true = torch.stack(self.y_true)
        # self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        # self.y_true = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list])
        # self.y_true = torch.tensor(self.y_true)

        if AF2model:

            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")
            
            graph_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_graph.pt"))
            #for prot_graph in graph_list_af:
            #    del prot_graph.x
            self.graph_list += graph_list_af
            
            y_true_af = [graph_dict[f'y_true_{task}'] for graph_dict in torch.load('data/processed/AF2_trainset.pt']
            
            self.y_true = np.concatenate([self.y_true, y_true_af],0)
            self.y_true = torch.tensor(self.y_true)

        if SWmodel and set_type=='val':

            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")

            graph_list_sw = torch.load(os.path.join(self.processed_dir, f"Swiss{set_type}_graph.pt"))

            self.graph_list += graph_list_sw

            y_true_sw = [graph_dict[f'y_true_{task}'] for graph_dict in torch.load('data/processed/swiss_valset.pt')]
            y_true_sw = torch.stack(y_true_sw)
            self.y_true = np.concatenate([self.y_true, y_true_sw],0)
            self.y_true = torch.tensor(self.y_true)

    def __getitem__(self, idx):

        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)


    def process(self):

        trainset_list = torch.load('data/processed/trainset.pt')
        train_graph_list = []
        for graph_dict in trainset_list:
            row, col = graph_dict['edge_index']
            edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
            seq_code = aa2idx(graph_dict['sequence'])
            seq_code = torch.IntTensor(seq_code)
            
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
            batch_tokens = batch_tokens.to(f"cuda:{self.device}")
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0].cpu().to(torch.float16)
            
            data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
            train_graph_list.append(data)
        
        val_graph_list = []
        valset_list = torch.load('data/processed/valset.pt')
        for graph_dict in valset_list:
            row, col = graph_dict['edge_index']
            edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
            seq_code = aa2idx(graph_dict['sequence'])
            seq_code = torch.IntTensor(seq_code)
        
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
            batch_tokens = batch_tokens.to(f"cuda:{self.device}")
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0].cpu().to(torch.float16)

            data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
            val_graph_list.append(data)

        test_graph_list = []
        testset_list = torch.load('data/processed/testset.pt')
        for graph_dict in testset_list:
            row, col = graph_dict['edge_index']
            edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
            seq_code = aa2idx(graph_dict['sequence'])
            seq_code = torch.IntTensor(seq_code)
        
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
            batch_tokens = batch_tokens.to(f"cuda:{self.device}")
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0].cpu().to(torch.float16)

            data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
            test_graph_list.append(data)
        
        torch.save(train_graph_list,'data/processed/train_graph.pt')
        torch.save(val_graph_list,'data/processed/val_graph.pt')
        torch.save(test_graph_list,'data/processed/test_graph.pt')

        if self.AF2model:
            trainset_list = torch.load('data/processed/AF2_trainset.pt')
            train_graph_list = []
            for graph_dict in trainset_list:
                row, col = graph_dict['edge_index']
                edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
                seq_code = aa2idx(graph_dict['sequence'])
                seq_code = torch.IntTensor(seq_code)
                
                batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
                batch_tokens = batch_tokens.to(f"cuda:{self.device}")
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33][0].cpu().to(torch.float16)
                
                data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
                train_graph_list.append(data)

            
            val_graph_list = []
            valset_list = torch.load('data/processed/AF2_valset.pt')
            for graph_dict in valset_list:
                row, col = graph_dict['edge_index']
                edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
                seq_code = aa2idx(graph_dict['sequence'])
                seq_code = torch.IntTensor(seq_code)
            
                batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
                batch_tokens = batch_tokens.to(f"cuda:{self.device}")
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33][0].cpu().to(torch.float16)

                data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
                val_graph_list.append(data)
            
            torch.save(train_graph_list,'data/processed/AF2train_graph.pt')
            torch.save(val_graph_list,'data/processed/AF2val_graph.pt')


class SwissDataSet(Dataset):
    def __init__(self, task, device):
        self.task = task
        self.processed_swissval = 'data/processed/Swissval_graph.pt'
        self.device = device
        
        if not os.path.exists(self.processed_swissval):
            self.model, alphabet = esm.pretrained.load_model_and_alphabet_local("/home/lhlai_pkuhpc/lustre3/zhgu/esm-1b/checkpoints/esm1b_t33_650M_UR50S.pt")
            self.batch_converter = alphabet.get_batch_converter()
            self.model = self.model.to(f"cuda:5")
            self.model.eval()
            self.process()
        
        self.graph_list = []
        for indx in range(7):
            self.graph_list += torch.load(f'data/processed/Swisstrain_graph{indx}.pt')
        self.y_true = [graph_dict[f'y_true_{task}'] for graph_dict in (torch.load('data/processed/swiss_trainset1.pt') + torch.load('data/processed/swiss_trainset2.pt'))[31471*0: 31471*7+31471]]
        self.y_true = torch.stack(self.y_true)
        
    def __getitem__(self, idx):
        #return_graph_list = []
        #for graph in self.graph_list:
        #    graph_cp = graph.clone()
        #    batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_cp.seq[:5], graph_cp.seq)])
        #    batch_tokens = batch_tokens.to(f"cuda:{device}")
        #    with torch.no_grad():
        #        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        #    token_representations = results["representations"][33][0].cpu().to(torch.float16)
        #    graph_cp.x = token_representations[1:len(sequence)+1]
        #    del graph_cp.sequence
        #    return_graph_list.append(graph_cp)
            
        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        
        return len(self.graph_list)
    
    def process(self):
        trainset_list = torch.load('data/processed/swiss_trainset1.pt') + torch.load('data/processed/swiss_trainset2.pt')
        train_graph_list = []
        for graph_dict in trainset_list:
            row, col = graph_dict['edge_index']
            edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
            seq_code = aa2idx(graph_dict['sequence'])
            seq_code = torch.IntTensor(seq_code)
            
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
            batch_tokens = batch_tokens.to(f"cuda:{self.device}")
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0].cpu().to(torch.float16)
            
            data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
            train_graph_list.append(data)
        
        val_graph_list = []
     
        valset_list = torch.load('data/processed/swiss_valset.pt')
        for graph_dict in valset_list:
            row, col = graph_dict['edge_index']
            edge_index = torch.LongTensor(torch.stack([row, col]))#torch.LongTensor([row, col])
            seq_code = aa2idx(graph_dict['sequence'])
            seq_code = torch.IntTensor(seq_code)
        
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(graph_dict['sequence'][:5], graph_dict['sequence'])])
            batch_tokens = batch_tokens.to(f"cuda:{self.device}")
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33][0].cpu().to(torch.float16)

            data = Data(x=token_representations[1:len(graph_dict['sequence'])+1], edge_index=edge_index, native_x=seq_code)
            val_graph_list.append(data)
        
        torch.save(train_graph_list,'data/processed/Swisstrain_graph.pt')
        torch.save(val_graph_list,'data/processed/Swissval_graph.pt')
