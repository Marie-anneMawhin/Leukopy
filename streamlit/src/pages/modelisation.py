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
        
        # Model structure and methodology
        st.subheader("""Model structure""")
        cont_1 = st.container()
        cont_1.markdown("""We replace the original VGG19 top layers with our own custom classification block :""")
        cont_1.image('./data/images/vgg19/vgg19_structure.png')
        
        cont_1 = st.container()
        cont_1.markdown("""To train this model, we adopt a **two-stage approach** :""")
        cont_1.markdown("""
                      + **transfer learning from ImageNet** for VGG19 : we only train the classification block layers,
                      + **fine-tuning of the 'block5'**.
        """)
                 
        cont_1.markdown("""
        In order to reduce a possible overfitting, we implement the following
        - **Data augmentation** (see below),
        - **Dropout layers** (after each Dense layer in the last block),
        - **Class weights** : during training, we penalize more when errors are committed on low population classes (e.g. PMY or BA),
        - **Early Stopping**.
        """)
            
        cont_1.markdown('''We use **adaptive learning rate** to accelerate the training, convergence 
                        and to improve generalization.''')
              
        
        # Preprocessing
        cont_2 = st.container()        
        cont_2.subheader("""Preprocessing""")
  
        cont_2.markdown("""The data flow through this pipeline during training. 
                      The same steps (except data augmentation) are applied to the test data.""")
                      
        """### display gif from local file in markdown"""
        file_ = open("./data/images/vgg19/preprocessing_1750_50.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        cont_2.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="vgg19 preprocessing gif">',unsafe_allow_html=True)
        cont_2.markdown('''''')
        
        # Results
        cont_3 = st.container()
        cont_3.subheader("""Main results""")
    
    
        cont_3.markdown('''
                        - ###### Loss and global accuracy :
        ''')
        cont_3.markdown(''' Global accuracy on training data : 95%. Global **accuracy on test** and validation data : **94%** ''')
        with cont_3.expander('Charts'):
            st.image('./data/images/vgg19/vgg19_training.png')
        
            st.markdown(''' There is no overfitting. 
                        The validation loss is much lower than the training loss 
                        because of dropout layers, data augmentation and class 
                        weights used during the training.''')


        cont_3.markdown('''
                        - ###### Classification report :
        ''')
        cont_3.markdown(''' 
        F1-Scores greater than 95% for all classes **except for : PMY, MY, MMY and BNE**.
        ''')
        with cont_3.expander('Charts'):
            st.image('./data/images/vgg19/vgg19_report.png')
        
        
        
        
        cont_3.markdown('''
                        - ###### Confusion matrix :
        ''')
        cont_3.markdown(''' 
        The main mistake commited by the model is **muddling up the different kind of neutrophiles** (matures and immatures).
        ''')
        with cont_3.expander('Charts'):
            st.image('./data/images/vgg19/vgg19_confuse.png')
               
                    
        
    if model_choice == "ViT-b16":
        pass