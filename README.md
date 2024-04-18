VARGG: An Integrated Framework Based on Vision Transformers and Graph Autoencoders to Advance Spatial Transcriptomics in the Fine Structure Analysis of Tissue Microenvironments


![model](https://github.com/w2260584531/VARGG-main/assets/140353599/bb83297c-0937-48c7-8740-d0f47735a89c)


## Requirements
```
python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3
```

The primary datasets used are as follows: 
1) The DLPFC (Dorsolateral Prefrontal Cortex) dataset, with detailed access and specifics to be provided in subsequent publications; 
2) Mouse Embryo Data, which can be downloaded from the China National GeneBank's Stomics platform (https://db.cngb.org/stomics/mosta);
3) Data pertaining to Glioblastoma, Breast Cancer, and Mouse Brain, available on the 10X Genomics website; and
4) Mouse Olfactory Bulb Data by Stereo-seq and other related spatial transcriptomics data, accessible via the spatialLIBD website (https://www.spatialomics.org/SpatialDB.).



## example
'''
import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from VARGG import running
import scanpy as sc



data_path = "/data/VARGG-main/data/DLPFC"

data_name = '151673' 

save_path = "/data/VARGG-main/Results" 

n_domains = 7 

process = running(save_path = save_path,pre_epochs = 1000, epochs = 1200, se_gpu = True)

adata = process._get_adata(platform="Visium", data_path=data_path, data_name=data_name)

adata = process._get_image_crop(adata, data_name=data_name) 

adata = process._get_augment(adata, spatial_type="KDTree", use_morphological=True)

graph_dict = process._get_graph(adata.obsm["spatial"], distType = "KDTree")

data = process._data_process(adata, pca_n_comps =200)

Vargg_embed = process._fit(
		data = data,
		graph_dict = graph_dict,)

adata.obsm["VARGG_embed"] = Vargg_embed

adata = process._get_cluster_data(adata, n_domains=n_domains, priori = True)


sc.pl.spatial(adata, color='VARGG_refine_domain', frameon=False,spot_size=150)
'''
