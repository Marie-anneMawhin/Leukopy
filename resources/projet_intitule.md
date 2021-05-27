# Blood cell classification

<!-- #region -->
### Cursus concerné : Data Scientist 

### Difficulté : 9/10

### Description détaillée

L’objectif de ce projet est d’identifier les différents types de cellules du sang à l’aide d'algorithmes de computer vision. La densité et l’abondance relative des cellules du sang dans le frottis est cruciale pour le diagnostic de nombreuses pathologies, comme par exemple pour la leucémie qui repose sur le ratio de lymphocytes. L’identification de leucocytes anormaux dans des pathologies telles que la leucémie pourrait compléter cette première partie.
Développer un outil capable d'analyser les cellules à partir de frottis sanguins pourrait faciliter le diagnostic de certaines pathologies mais aussi être utilisé à but de recherche.

### Data : 
A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data Brief. 2020 Jun; 30: 105474.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/
Article: https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub


Size :  17092 images de microscopie d’Individus sains anonymisés (MGG stain)
(360 x 363 pixels) as jpg

Labels: réalisé par des cliniciens pathologistes
8 types de cellules 
(neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts and platelets or thrombocytes)

Data de patient leucémique pour entraîner le modèle sur des leucocytes malades.

https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl
https://www.kaggle.com/nikhilsharma00/leukemia-dataset

Article:https://www.sciencedirect.com/science/article/abs/pii/S0169260721000742?via%3Dihub




### Possibilité d'enrichissement : 
Mes propres données
GAN? Transformation classique (rotation…)
https://public.roboflow.com/object-detection/bccd
https://www.kaggle.com/paultimothymooney/blood-cells
google image 

### Modélisation
Transfer learning : CNN
Classification des leucocytes
Identification de leucocytes anormaux
segmentation
<!-- #endregion -->

```python

```
