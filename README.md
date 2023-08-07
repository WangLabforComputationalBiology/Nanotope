# Nanotope
**Nanotope: A Graph Neural Network-Based Model for Nanobody paratope prediction**

Nanobodies, from camelid species' immune system, are small artificial antibodies obtained by isolating antigen-binding proteins. Their high specificity enables versatile applications. Experimental nanobody paratope determination is costly and time-consuming, while computational methods lack accuracy. To improve prediction, we propose Nanotope, a structure-based model using graph neural networks and a pre-trained language model for precise paratope prediction.

# Install

**Clone the repo**

```
git clone https://github.com/Wo-oh-oh-ooh-oh/Nanotope
cd Nanotope
```

**Create a virtual env**

```
conda create --name Nanotope python=3.9
```

**install**

```
pip install .
```

If you encounter an error while installing the `torch_geometric` package, please refer to the following [link](https://zhuanlan.zhihu.com/p/519168089) for troubleshooting

# Usage

If you want to predict a single nanobody, please refer to the method in **Nanotope/example/predict.ipynb** for prediction. If you want to perform batch prediction, please use **Nanotope/data/pre_data.py** to process a large batch of data and store it in memory before making predictions. Both of these prediction methods require **nanobody PDB files** and corresponding **heavy-chain IDs.**

