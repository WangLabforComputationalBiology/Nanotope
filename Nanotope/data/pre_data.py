import pandas as pd
import numpy as np
from antiberty import AntiBERTyRunner
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_cluster import knn
from torch_geometric.data import InMemoryDataset
from utils import set_coord

class NanotopeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NanotopeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['tcm_dataset.pt']

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass
    
    def make_data(self,df):
        k = 32 # The number of edges is 32
        dataset = []
        for seq,coord in (zip(df['sequence'],df['coord'])):
            seq = [seq]
            size = len(seq[0])
            coord = torch.tensor(coord)

            # get seq embedding by Antiberty model 
            Antiberty = AntiBERTyRunner()
            embeddings = Antiberty.embed(seq)[0][1:-1]

            #padding if len(seq)<140, using zero vetor [0,...,0]
            if size<140:
                pad = torch.zeros((140-size),512).cuda()
                embeddings =torch.cat([embeddings,pad],dim=0)

            # construct KNN edges   
            edge_index = knn(coord,coord,k = k)

            # construct graph data
            data = Data(x = embeddings,edge_index=edge_index,mask=size)
            dataset.append(data)
            
        return dataset

    def process(self):
        
        # Read data into huge `Data` list. using a parquet data structure
        Nano = pd.read_parquet(r'E:\608\paratope\data\parquet\Nanobody_set.parquet')
        Nano = set_coord(Nano,r'E:\608\paratope\data\PDB\nano')
   
        Nano = Nano.reset_index(drop = True)
        data_list = self.make_data(Nano)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ =='__main__':
    Dataset = NanotopeDataset('./GraphData')