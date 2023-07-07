from Bio.PDB.PDBParser import PDBParser
import numpy as np
import glob
import sys, os
import pickle as pkl
import argparse 
import torch
from torch_geometric.data import Data, Batch
from utils import protein_graph
from network import CL_protNET
from utils import load_GO_annot
import esm

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False
        
p = argparse.ArgumentParser()
p.add_argument('--device', type=str, default='', help='')
p.add_argument('--task', type=str, default='bp', choices=['bp','mf','cc'], help='gene ontoly task')
p.add_argument('--pdb', type=str, default='', help='Input the query pdb file')
p.add_argument('--esm1b_model', type=str, default='', help='The path of esm-1b model parameter.')
p.add_argument('--only_pdbch', default=False, type=str2bool, help='To use model parameters trained only on PDBch training set')
p.add_argument('--prob', default=0.5, type=float, help='Output the function with predicted probility > 0.5 .')
args = p.parse_args()

_, goterms, gonames, _ = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


restype_3to1 = {v: k for k, v in restype_1to3.items()}

pdb = args.pdb
parser = PDBParser()

struct = parser.get_structure("x", pdb)
model = struct[0]
chain_id = list(model.child_dict.keys())[0]
chain = model[chain_id]
Ca_array = []
sequence = ''
seq_idx_list = list(chain.child_dict.keys())
seq_len = seq_idx_list[-1][1] - seq_idx_list[0][1] + 1

for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1]+1):
    try:
        Ca_array.append(chain[(' ', idx, ' ')]['CA'].get_coord())
    except:
        Ca_array.append([np.nan, np.nan, np.nan])
    try:
        sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]
    except:
        sequence += 'X'

#print(sequence)
Ca_array = np.array(Ca_array)
    
resi_num = Ca_array.shape[0]
G = np.dot(Ca_array, Ca_array.T)
H = np.tile(np.diag(G), (resi_num,1))
dismap = (H + H.T - 2*G)**0.5

device = f'cuda:{args.device}'

esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.esm1b_model)

batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)

esm_model.eval()

batch_labels, batch_strs, batch_tokens = batch_converter([('tmp', sequence)])
batch_tokens = batch_tokens.to(device)
with torch.no_grad():
    results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33][0].cpu().numpy().astype(np.float16)
    esm_embed = token_representations[1:len(sequence)+1]

row, col = np.where(dismap <= 10)
edge = [row, col]
graph = protein_graph(sequence, edge, esm_embed)
batch = Batch.from_data_list([graph])

if args.task == 'bp':
    output_dim = 1943
elif args.task == 'mf':
    output_dim = 489
elif args.task == 'cc':
    output_dim = 320

model = CL_protNET(output_dim).to(device)
if args.only_pdbch:
    model.load_state_dict(torch.load(f'model/model_{args.task}CL.pt',map_location=device))
else:
    model.load_state_dict(torch.load(f'model/model_{args.task}CLaf.pt',map_location=device))
model.eval()
with torch.no_grad():
    y_pred = model(batch.to(device)).cpu().numpy()
func_index = np.where(y_pred > args.prob)[1]
if func_index.shape[0] == 0:
    print(f'Sorry, no functions of {args.task.upper()} can be predicted...')
else:
    print(f'The protein may hold the following functions of {args.task.upper()}:')
    for idx in func_index:
        print(f'Possibility: {round(float(y_pred[0][idx]),2)} ||| Functions: {goterms[args.task][idx]}, {gonames[args.task][idx]}')
