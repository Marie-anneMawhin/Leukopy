# Leukopy 🩸

Classification of white blood cells.


## Context
The identification and classification of leukocytes, platelets and erythrocytes is crucial in the diagnosis of several hematological diseases, such as infectious diseases, regenerative anaemia or leukemia.

# Table of content
 * [Repository structure](#repository-structure)
 * [Context](#context)
 * [Data](#data)
 * [Contributors](#contributors)

## Repository structure

```bash
├── data
├── demo
├── notebooks
│   ├── EDA
│   └── model
│       ├── efficientNet
│       ├── lenet
│       ├── vgg16
│       ├── vgg19*
│       ├── ViT
│       └── Xception
└── streamlit
```
(*) **VGG19** Version for all data available and executable on Drive: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1rjNVlA6UtFM7uT9kQFlePRRp4cTB0kcl/view?usp=sharing)  

```data```: `.csv` indicating `path`, `label`, `size` of images

```notebooks``` : jupyter notebooks in the folders model (exploration of various models) and EDA (data exploration).

```streamlit```: streamlit app.

## Usage

The datasets can be downloaded using the the links provided in their description: see [Data](#data) and place in the data folder. 

For the barcelone and the merge dataset: the notebook in `import` can generate the dataframe in `data`.

The barcelone dataset *PBC_dataset_normal_Barcelona* can be used on its own.  
The merge data should follow the following structure and be merged following the table in [Data](#data):
```bash
├── BA
├── BNE
├── EO
├── ERB
├── LY
├── MMY
├── MO
├── MY
├── PLT
├── PMY
└── SNE
```

## Data

### 1. Normal peripheral blood cells

#### *PBC_dataset_normal_Barcelona*

A publicly available [dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) of more than 17000 images of blood leukocytes from blood smears of healthy donors stained with MGG.<sup>[1](#footnote1)</sup>

Image size: 360 x 363 px, naming: <label>_<anonymised id>.jpg

8 or 11 classes of white blood cells:
- neutrophils (segmented) - SNE
- eosinophils - EO
- basophils - BA
- lymphocytes - LY
- monocytes - MO
- immature granulocytes (metamyelocytes, myelocytes, promyelocytes) and band - IG or separated - MMY, MY, PMY, BNE
- platelets - PLATELET
- erythroblasts - ERB
<br />
 
 
 #### *PBC_dataset_normal_Raabin*
A publicly available [dataset](https://raabindata.com/free-data/) around 15,000 images<sup>[3](#footnote3)</sup>. All samples are healthy except for a few basophils imaged from a leukemia patient and come from three laboratories in Iran: Razi Hospital in Rasht, Gholhak Laboratory, Shahr-e-Qods Laboratory and Takht-e Tavous Laboratory in Tehran.
 
 5 classes of white blood cells:
- neutrophils - NE (not include in merged dataset)
- eosinophils - EO
- basophils - BA
- lymphocytes - LY
- monocytes - MO

Image size: 300 x 300 px, naming: in class_label_CVB.csv, labels (1- 5) corresponds respectively to neutrophil, lymphocyte, monocyte, eosinophil and basophil.
 
<br />
 

### 2.PBS_dataset_AML_Munich

A publicly available [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958). The data corresponds to 100 patients diagnosed with Acute Myeloid Leukemia at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of hematological malignancy.<sup>[4](#footnote4)</sup>
- BAS Basophil
- EBO Erythroblast
- EOS Eosinophil
- KSC Smudge cell : NOT USED
- LYA Lymphocyte (atypical) : NOT USED
- LYT Lymphocyte (typical)
- MMZ Metamyelocyte
- MOB Monoblast : NOT USED
- MON Monocyte
- MYB Myelocyte
- MYO Myeloblast: NOT USED
- NGB Neutrophil (band)
- NGS Neutrophil (segmented)
- PMB Promyelocyte (bilobled) : NOT USED
- PMO Promyelocyte

<a name="footnote1">1.</a> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)

<a name="footnote2">2.</a> Fast and Robust Segmentation of White Blood Cell Images by Self-supervised Learning. [Micron
Volume 107, April 2018, Pages 55-71](https://doi.org/10.1016/j.micron.2018.01.010)
 
 <a name="footnote3">3.</a> Raabin-WBC: a large free access dataset of white blood cells from normal peripheral blood. [bioRxiv, 5 (2021))](https://www.biorxiv.org/content/10.1101/2021.05.02.442287v4)

<a name="footnote4">4.</a> Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. [Nature Machine Intelligence volume 1, pages 538–544 (2019)](https://www.nature.com/articles/s42256-019-0101-9)
 
 
 ### 3.Merged final dataset
A merge dataset consisting of the 3 datasets described above and 11 classes:

|Cell type|Code|Barcelona|Raabin|Munich|
|---------|----|---------|------|------|
|neutrophils (segmented)| SNE|X| |X|
|eosinophils|              EO|X|X|X|
|basophils|                BA|X|X|X|
|lymphocytes|              LY|X|X|X|
|monocytes|                MO|X|X|X|
|metamyelocytes|          MMY|X| |X|
|myelocytes|               MY|X| |X|
|promyelocytes|           PMY|X| |X|
|band neutrophils|        BNE|X| |X|
|platelets|               PLT|X| | |
|erythroblasts|           ERB|X| |X|
 


## Contributors
 
Mathieu Sarrat

Marie-Anne Mawhin
 
Laleh Ravanbod

Yahia Bouzoubaa

