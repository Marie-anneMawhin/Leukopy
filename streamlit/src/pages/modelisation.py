import streamlit as st
import base64

from importlib import reload


model_list = ["VGG16", "VGG19", "ViT-b16"]


def write():
    st.title("Modelisation")

    model_choice = st.selectbox("Select a model", options=model_list)
    
    if model_choice == "VGG16":
        pass

    if model_choice == "VGG19":
        
        cont = st.container()
        

        cont.subheader("""Model structure :""")
        cont.markdown("""We replace the original VGG19 top layers with our own custom classification block.""")
        cont.image('./data/images/vgg19/vgg19_structure.png', 
                   caption = "We use the five convolutional blocks of VGG19 ... \n and connect them to the following layers :")

        cont.image('./data/images/vgg19/vgg19_classblock.png', width = 150)
        
        cont.markdown("""To train this model, we adopt a **two-stage approach** :""")
        cont.markdown("""
                      + **transfer learning from ImageNet** for VGG19 : 
                      we only train the classification block layers,
                      + **fine-tuning of the 'block5'** : we train block5 layers (4 Conv2D layers) 
                      and adjust the weights of the classification block </br>""", unsafe_allow_html = True)
            
        cont.markdown("""In order to reduce a possible overfitting, we implement the following :""")
        cont.markdown("""+ **Data augmentation** (see below),""")
        cont.markdown("""+ **Dropout layers** (after each Dense layer in the last block),""")
        cont.markdown("""+ **Class weights** : during training, we penalize more when errors are committed on low population classes (e.g. PMY or BA)
                        """)
                        
        
                
        cont.subheader("""Preprocessing :""")
  
        """### display gif from local file in markdown"""
        file_ = open("./data/images/vgg19/preprocessing_1750_50.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        cont.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="vgg19 preprocessing gif">',unsafe_allow_html=True)
        
        cont.subheader("""Main results :""")
        with cont.expander("""Main results :"""):
            
            st.markdown("**Loss and global accuracy :**")
            st.image('./data/images/vgg19/vgg19_training.png')
            
            st.markdown("Global accuracy on test data : **0.94**")
            
            st.markdown("**Classification report :**")
            st.image('./data/images/vgg19/vgg19_report.png')
            
            st.markdown("**Confusion matrix :**")    
            st.image('./data/images/vgg19/vgg19_confuse.png', width = 500)
            
                      
        pass
        
    if model_choice == "ViT-b16":
        pass