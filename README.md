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
1) The DLPFC (Dorsolateral Prefrontal Cortex) dataset, accessible within the spatialLIBD package (http://spatial.libd.org/spatialLIBD); 
2) 2) Data pertaining to Glioblastoma, Breast Cancer, and Mouse Brain, available on the 10X Genomics website (https://support.10xgenomics.com/spatial-gene-expression/datasets); 
3) Mouse Embryo Data, which can be downloaded from the China National GeneBank's Stomics platform (https://db.cngb.org/stomics/mosta); 
4) Slide-seqV2 datasets are available at the Broad Institute Single Cell Portal at https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary; 
5) The processed Stereo-seq data from mouse olfactory bulb tissue is accessible on https://github.com/JinmiaoChenLab/SEDR_analyses; 
6) The MERFISH dataset is available from https://github.com/zhengli09/BASS-Analysis.  


## Pre-trained VIT model download link
'''
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
'''


## Tutorial
A Jupyter Notebook of the tutorial is accessible from :
https://github.com/w2260584531/VARGG-main/tree/main/tutorial
