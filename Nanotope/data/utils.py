import pandas as pd
import os
from Bio.PDB import *
import numpy as np

import torch
from antiberty import AntiBERTyRunner
from torch_cluster import knn
from torch_geometric.data import Data
import warnings

import warnings
warnings.filterwarnings("ignore")


NB_MAX_LENGTH = 140
NB_CHAIN_ID = "H"
BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"]
OUTPUT_SIZE = len(BACKBONE_ATOMS) * 3

def get_seq_aa(pdb_file, chain_id):
    # load model
    chain = PDBParser(QUIET=True).get_structure(pdb_file, pdb_file)[0][chain_id]

    aa_residues = []
    seq = ""

    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'): # Not amino acid
            continue
        elif aa == "UNK":  # unkown amino acid
            seq += "X"
        else:
            seq += Polypeptide.three_to_one(residue.get_resname())
        aa_residues.append(residue)

    return seq, aa_residues

def generate_coord(pdb_file,chain_id):  
    # get seq and aa residues
    seq, aa_residues = get_seq_aa(pdb_file, chain_id)

    # turn into backbone + CB xyz matrix
    xyz_matrix = np.zeros((NB_MAX_LENGTH, OUTPUT_SIZE))
    for i in range(len(aa_residues)):
        for j, atom in enumerate(BACKBONE_ATOMS):
            if not (atom=="CB" and seq[i] == "G"):
                xyz_matrix[i][3*j:3*j+3] = aa_residues[i][atom].get_coord()

    return xyz_matrix[:,3:6]

def set_coord(df,pdb_dir):
    
    empty_col = pd.Series('',index=df.index)
    df['coord'] = empty_col

    for index, row in df.iterrows():
        pdb_name = row['PDB']
        chain_id = row['chain_id']
        
        # use pdb and chain id to get aa coordinates 
        pdb_coord = generate_coord(os.path.join(pdb_dir,pdb_name),chain_id)
        df.at[index,'coord'] = pdb_coord

    return df


def make_data(df,k):

    dataset = []
    for index, row in df.iterrows():

        seq = list(row['sequence'])
        size = len(seq[0])
        label = torch.tensor(np.where(row['paratope_labels'] == 'P', 1, 0))
        coord = torch.tensor(row['coord'])

        Antiberty = AntiBERTyRunner()
        embeddings = Antiberty.embed(seq)[0][1:-1]
        if size<140:
            pad = torch.zeros((140-size),512).cuda()
            embeddings =torch.cat([embeddings,pad],dim=0)


        edge_index = knn(coord,coord,k = k)
        data = Data(x = embeddings, y= label,pos =coord,edge_index=edge_index,mask=size)
        dataset.append(data)
    
    return dataset



if __name__ =='__main__':
    train = pd.read_parquet(r'E:\608\NanoNet-main\paratope\data\parquet\train.parquet')
    test = pd.read_parquet(r'E:\608\NanoNet-main\paratope\data\parquet\test.parquet')
    val = pd.read_parquet(r'E:\608\NanoNet-main\paratope\data\parquet\val.parquet')

    val = set_coord(val,r'E:\608\NanoNet-main\paratope\data\PDB\val')
    train = set_coord(train,r'E:\608\NanoNet-main\paratope\data\PDB\train')
    test = set_coord(test,r'E:\608\NanoNet-main\paratope\data\PDB\test')

    val = val.reset_index(drop = True)
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)