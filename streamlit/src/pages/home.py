import streamlit as st


def write():
    st.title("Leukopy 🩸 - blood cell classifier")

    st.markdown("""
    ## Context
    The identification and classification of leukocytes, platelets and erythrocytes 
    is crucial in the **diagnosis** of several haematological diseases, 
    such as infectious diseases or leukaemia.
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        Visual and qualitative evaluation of blood smears is often necessary for diagnosis.
        However, the manual identification of peripheral blood cells is **challenging**, **time consuming**, 
        prone to **error** and requires the presence of a **trained specialist**.  
        """)

    with col2:
        st.image('src/data/images/Blood-smear-prep_HEADER.png')

    st.markdown("""
        ## Problematic
        **Computer-assisted analysis** of blood smears and identification of 
        abnormal cells provided crucial assistance to researchers and clinicians.
        """)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image('src/data/images/segmentation.png')

    with col2:
        st.markdown("""
        Traditionally, white blood cell recognition systems have relied on 
        segmentation, component separation, feature extraction, and 
        classification of white blood cells by shallow machine learning models.
        """)

    st.markdown("""
    This type of systems do not **generalise** well, in particular because of the 
    variability of Romanowsky staining and acquisition systems along with the significant
    requirements of its design in terms of pre-processing and feature extraction.
    """)

    st.markdown(""" ## Aim 
    <p style='font-size: 22px;'> The main objective of this study is to develop 
    a <b>deep neural network</b> capable of classifying healthy blood cells
     into 8 or 11 classes.
     </p> """, unsafe_allow_html=True)

    st.markdown("""
    ### _Description of cell types_
    We are going to focus on the following cells:
    - neutrophils (segmented) - SNE
    - eosinophils - EO
    - basophils - BA
    - lymphocytes - LY
    - monocytes - MO
    - platelets - PLATELET
    - erythroblasts - ERB
    - immature (metamyelocytes, myelocytes, promyelocytes) and band neutrophils - IG or separated - MMY, - MY, - PMY, and - BNE

    """
                )

    hemato = open('src/data/images/hemato.mp4', 'rb')
    hemato_bytes = hemato.read()
    st.video(hemato_bytes)

    st.markdown(""" 
    ##
    Each cells shows different characteristics such as:
    - the colour of their granules (pink/orange, purple, lilac)
    - the number of nuclear lobes
    - the shape of the nucleus
    - the colour of cytoplasm ...


    ## Data

    #### 1.PBC_dataset_normal_Barcelona

    A publicly available [dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) of more than 
    17000 images of blood leukocytes from blood smears of healthy donors stained with MGG.<sup>[1](#footnote1)</sup>
    
    #### 2.PBC_dataset_normal_Raabin
    A publicly available [dataset](https://raabindata.com/free-data/) around 15,000 images. 
    All samples are healthy except for a few basophils imaged from a leukaemia patient and come from three laboratories in Iran: 
    Razi Hospital in Rasht, Gholhak Laboratory, Shahr-e-Qods Laboratory and Takht-e Tavous Laboratory in Tehran.<sup>[2](#footnote2)</sup>
    
    #### 3.PBS_dataset_AML_Munich
    A publicly available [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958). The data corresponds to 100 patients diagnosed 
    with Acute Myeloid Leukemia at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of haematological malignancy.<sup>[3](#footnote3)</sup>
    


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

    </br>
    </br>

   <a name="footnote1">1.</a> *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)*
 
    <a name="footnote2">2.</a> *Raabin-WBC: a large free access dataset of white blood cells from normal peripheral blood. [bioRxiv, 5 (2021)](https://www.biorxiv.org/content/10.1101/2021.05.02.442287v4)*

    <a name="footnote3">3.</a> *Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. [Nature Machine Intelligence volume 1, pages 538–544 (2019)](https://www.nature.com/articles/s42256-019-0101-9)*
    """, unsafe_allow_html=True)
