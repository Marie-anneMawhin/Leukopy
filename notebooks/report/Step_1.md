<!-- #region -->
# A/Context


This project aims at identifying and classifying white blood cells, including platelets and erythroblasts, A broader approach will aim at classifying other abnormal cells related to acute myeloid leukemia.

<!-- #endregion -->

<!-- #region -->
## 1.Motivation
Several pathologies can be detected haematologically, as for example the identification and classification of leukocytes, platelets and erythrocytes is crucial in the diagnosis of regenerative anaemia. While several automatic counting instruments such as flow cytometer or hemocytometer provides information on counts, the qualitative assessment of cell by imaging is often necessary. In clinics, blood smear are indicated in case of AIDS, sepsis, organ failure, unexplained leukocytosis, anaemias, or suspected leukaemia.  
Manual determination of peripheral blood cells is time-consuming and expensive. Moreover, it requires the need for trained specialist. This is difficult to obtain in a research context, where blood smear are often used to assess the quality of a cell preparation. Finally, manual assessment is prone to erroneous results.  


Computer-aided analysis peripheral blood smear and identification of abnormal cells would provide a crucial assistance for research and clinicians. Traditionally, white blood cell recognition systems rely on segmentation, separation of components, feature extraction and classification of white cells. This type of method is difficult to generalise and its design is truly challenging and here the use of deep learning to meet the require performance makes all its sense.

<!-- #endregion -->

<!-- #region -->
## 2.Type white blood cells

Circulating leukocytes are divided into 5 main classes;
- granulocytes: neutrophils, basophils and eosinophils
- monocytes
- lymphocytes
- erythrocytes (red blood cell)
- platelets or thrombocytes

Our datasets of healthy cells also includes immature cells that have emerge early from the bone marrow.
- erythoblast or nucleated RBC
- band neutrophils (which will be included with mature neutrophils even though they are immature)
- immature granulocytes (which can be kept as one group as labelling is extremely subjective)



The following diagram shows how these cells looks like following the classical staining (May Grunwald Giemsa or MGG) used in clinics.
<!-- #endregion -->

![image.png](https://upload.wikimedia.org/wikipedia/commons/6/69/Hematopoiesis_%28human%29_diagram.png)


## 3. Identification and function of leukocytes


![leuko](images/1914_Table_19_3_1.jpg)


# B/Data
||Healthy Blood|APL vs AML|
---|---|---
**origin**|Anonymysed healthy blood smears stained with MGG <br />(automatic stainer, QC)|Patients APL or AML at John Hopkins. <br />106 (retrospective for AML)
**size**|17092 RGB images of individual cells | APL = 22 patients <br /> AML = 60 patients <br />Validation : APL = 12 patients<br /> AML = 12 patients
**type**|.jpg| .jpg  
**size**|360 x 363 px| 360 x 363 px
**annotation**| CellaVision <br /> expert pathologist| CellaVision <br />FISH positive (molecular technique) for APL
**cell type**|8 classes <br /> neutrophils (segmented and band) <br /> eosinophils <br /> basophils<br />lymphocytes<br /> monocytes<br />immature granulocytes (metamyelocytes, myelocytes and promyelocytes)<br /> erythroblasts<br />platelets|Blasts<br />neutrophils<br />eosinophils<br /> basophils<br />lymphocytes monocytes<br />immature granulocytes (metamyelocytes, myelocytes and promyelocytes)<br />erythroblasts<br />platelets<br />giant thrombocyte<br />variant lymphocyte (large)<br />smudge cells (cancer)

**references**  
[normal cell - ncbi](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/|https://ash.confex.com/ash/2020/webprogram/Paper135836.html)  
[AML/APL - springernature](https://springernature.figshare.com/articles/dataset/Data_record_for_the_article_Deep_learning_for_diagnosis_of_Acute_Promyelocytic_Leukemia_via_recognition_of_genomically_imprinted_morphologic_features/14294675)











<!-- #region -->
# C/ Bias

## 1.Normal blood cell
Identified in EDA: platelets are small and thus the image is brighter.
Other potential biais: 
- generalisation as all analysis are done on one staining machine, one lab and one analyser.

## 2.Common to both datasets
- doublets
- staining precipitate and other artifacts
- RBC density


## 3.Acute myelocytic leukemia
- Few cells are blasts, so class imbalance
<!-- #endregion -->

# D/ Limitations

Our model will not be able to:
- recognise abnormalities, e.g. sickle cell, parasite, schistocytes….
- generalised to whole slide
- as there is no segmentation, no characteristic of cell shape or size will be extracted


```python

```
