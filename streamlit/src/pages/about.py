import streamlit as st


def write():
    st.header('About')

    st.markdown("""### Contributors

    Mathieu Sarrat

    Marie-Anne Mawhin

    Laleh Ravanbod""")

    st.markdown("""

    ### Image references

   *White blood cells identification system based on convolutional deep .... https://pubmed.ncbi.nlm.nih.gov/29173802/. *  
   
   *Hematopoiesis (human) - [wikipedia](https://fr.wikipedia.org/wiki/Fichier:Hematopoiesis_(human)_diagram.png)*  
   
   *Transformers for Image Recognition at Scale [Google AI blog](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)*
   
   *Neural Abstractive Text Summarization and Fake News Detection https://arxiv.org/abs/1904.00788*  

   *The Ontogeny of a Neutrophil: Mechanisms of Granulopoiesis and Homeostasis. DOI: https://doi.org/10.1128/MMBR.00057-17*  

    """, unsafe_allow_html=True)
