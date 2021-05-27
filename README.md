# Leukopy 🩸

Classification of white blood cells.

## Context
The identification and classification of leukocytes, platelets and erythrocytes is crucial in the diagnosis of several hematological diseases, such as infectious diseases or regenerative anaemia.
A broader approach will aim at classifying other abnormal cells related to acute leukemia.

## Data

### Normal peripheral blood cells:
A publicly available [dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) of more than 17000 images of blood leukocytes from blood smears of healthy donors stained with MGG.<sup>[1](#footnote1)</sup>

Image size: 360 x 363 px, naming: <label>_<anonymised id>.jpg

8 types of white blood cells:
- neutrophils (segmented) - SNE
- eosinophils - EO
- basophils - BA
- lymphocytes -LY
- monocytes -MO
- immature granulocytes (metemyelocytes, myelocytes, promyelocytes) and band - IG
- platelets - PLATELET
- erythroblasts - ERB


### APL, AML

A publicly available dataset [dataset](https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl). The data corresponds to 106 patients with identified Acute Promyelocytic Leukemia (APL) or Acute Myelocytic Leukemia (AML).<sup>[2](#footnote1)</sup>



<a name="footnote1">1.</a> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)


<a name="footnote1">2.</a> Deep Learning for Distinguishing Morphological Features of Acute Promyelocytic Leukemia. [Blood, 136, Supplement 1, 11 2020](https://doi.org/10.1182/blood-2020-135836)


## Repository structure


```notebooks``` : jupyter notebooks  

```data```

## Contributors

Laleh Ravanbod

Marie-Anne Mawhin

Mathieu Sarrat

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
