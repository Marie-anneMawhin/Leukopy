# Leukopy 🩸

Classification of white blood cells.

## Context
The identification and classification of leukocytes, platelets and erythrocytes is crucial in the diagnosis of several hematological diseases, such as infectious diseases or regenerative anaemia.
A broader approach will aim at classifying other abnormal cells related to acute leukemia.

## Data

### Normal peripheral blood cells:
A publicly available [dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) of more than 17000 images of blood leukocytes from blood smears of healthy donors stained with MGG.<sup>[1](#footnote1)</sup>

8 types of white blood cells:
- neutrophils
- eosinophils
- basophils
- lymphocytes
- monocytes
- immature granulocytes (metemyelocytes, myelocytes, promyelocytes)
- platelets
- erythroblasts

<a name="footnote1">1.</a> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)

### APL, AML

[dataset](https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl)

## Repository structure


```notebooks``` : jupyter notebooks  

```data```

## Drive
[drive](https://drive.google.com/drive/folders/14rP7TLwCbGqefV5b8lAWYhFASDMVm_bo)

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
