# Leukopy 🩸

Classification of white blood cells.

## Context
The identification and classification of leukocytes, platelets and erythrocytes is crucial in the diagnosis of several hematological diseases, such as infectious diseases or regenerative anaemia.
A broader approach will aim at classifying other abnormal cells related to acute leukemia.

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
- immature granulocytes (metemyelocytes, myelocytes, promyelocytes) and band - IG or separated - MMY, MY, PMY, BNE
- platelets - PLATELET
- erythroblasts - ERB
<br />
 
#### *PBC_dataset_normal_CVB*
A publicly available small [dataset](https://data.mendeley.com/datasets/w7cvnmn4c5/1) from [CellaVision Blog](http://blog.cellavision.com/) of 100 images of healthy donors blood smears <sup>[2](#footnote1)</sup>

Image size: 300 x 300 px, naming: in class_label_CVB.csv, labels (1- 5) corresponds respectively to neutrophil, lymphocyte, monocyte, eosinophil and basophil.
 
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
- KSC Smudge cell
- LYA Lymphocyte (atypical)
- LYT Lymphocyte (typical)
- MMZ Metamyelocyte
- MOB Monoblast
- MON Monocyte
- MYB Myelocyte
- MYO Myeloblast
- NGB Neutrophil (band)
- NGS Neutrophil (segmented)
- PMB Promyelocyte (bilobled)
- PMO Promyelocyte

<a name="footnote1">1.</a> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)

<a name="footnote2">2.</a> Fast and Robust Segmentation of White Blood Cell Images by Self-supervised Learning. [Micron
Volume 107, April 2018, Pages 55-71](https://doi.org/10.1016/j.micron.2018.01.010)
 
 <a name="footnote3">3.</a> Raabin-WBC: a large free access dataset of white blood cells from normal peripheral blood. [bioRxiv, 5 (2021))](https://www.biorxiv.org/content/10.1101/2021.05.02.442287v4)

<a name="footnote4">4.</a> Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. [Nature Machine Intelligence volume 1, pages 538–544 (2019)](https://www.nature.com/articles/s42256-019-0101-9)


## Repository structure


```notebooks``` : jupyter notebooks  

```data```: csv and pkl indicating path, label, size of images

## Contributors
 
Mathieu Sarrat

Marie-Anne Mawhin
 
Laleh Ravanbod

Yahia Bouzoubaa
 
## How to update new packages

### > I want to install new package

```
cd PROJECT_ROOT_DIRECTORY
pip install PACKAGE_NAME
pip-chill > requirements.txt
git add requirements.txt
git commit -m "COMMENT"
git pull
git push
```
### > I want to update packages according to other's changes
```
cd PROJECT_ROOT_DIRECTORY
git pull
pip install -r requirements.txt
```
**Note** The commands in capital letters are placeholders.
