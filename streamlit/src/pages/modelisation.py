import streamlit as st

from PIL import Image
from importlib import reload

import utils.common as common
reload(common)

model_list = ["VGG16+SVM", "VGG19", "ViT-b16"]


def write():
    st.title("Modelisation")

    # Model structure and methodology
    cont_1 = st.container()

    cont_1.markdown("""
        In order to reduce a possible overfitting, we implement the following for all models:
        - **Data augmentation** (see below),
        - **Dropout layers** (after each Dense layer in the last block, for CNNs),
        - **Class weights**: during training, we penalize more when errors are committed on low population classes (e.g. PMY or BA),
        - **Early Stopping**.
        """)

    cont_1.markdown('''We use **adaptive learning rate** to accelerate the training, convergence
                        and to improve generalization.''')

    model_choice = st.selectbox("Select a model", options=model_list)

    # VGG16 + SVM
    if model_choice == "VGG16+SVM":

        cont_2 = st.container()
        cont_2.image('./data/images/vgg16svm/modeles.png')

        cont_4 = st.container()
        cont_4.image('./data/images/vgg16svm/algorithmes.png')

        cont_3 = st.container()
        cont_3.image('./data/images/vgg16svm/filtre.png')
        
    # VGG19
    if model_choice == "VGG19":

        # Architecture
        cont_2 = st.container()

        cont_2.subheader("""Model architecture""")

        cont_2.markdown(
            """We replace the original VGG19 top layers with our own custom classification block:""")
        cont_2.image('./data/images/vgg19/vgg19_structure.png')

        cont_2.markdown("""
        To train this model, we adopt a **two-stage approach**:
        - **transfer learning from ImageNet** for VGG19: we only train the classification block layers,
        - **fine-tuning of the 'block5'**.
        """)

        # Preprocessing
        cont_3 = st.container()
        cont_3.subheader("""Preprocessing""")

        cont_3.markdown("""The data flow through this pipeline during training.
                        The same steps (except data augmentation) are applied to the test data.""")

        """### display gif from local file in markdown"""
        common.display_md_gif(path="./data/images/vgg19/preprocessing_1750_50.gif",
                              container=cont_2,
                              alt_text='vgg19 preprocessing gif')
        cont_3.markdown('''''')

        # Results
        cont_4 = st.container()
        cont_4.subheader("""Main results""")

        cont_4.markdown('''
                        - ###### Loss and global accuracy:
        ''')
        cont_4.markdown(
            ''' Global accuracy on training data: 95%. Global **accuracy on test** and validation data: **94%** ''')
        with cont_4.expander('Chart'):
            st.image('./data/images/vgg19/vgg19_training.png')

            st.markdown(''' There is no overfitting.
                        The validation loss is much lower than the training loss
                        because of dropout layers, data augmentation and class
                        weights used during the training.''')

        cont_4.markdown('''
                        - ###### Classification report:
        ''')
        cont_4.markdown('''
        F1-Scores greater than 95% for all classes **except for: PMY, MY, MMY and BNE**.
        ''')
        with cont_4.expander('Table'):
            st.image('./data/images/vgg19/vgg19_report.png')

        cont_4.markdown('''
                        - ###### Confusion matrix:  
        
        The model is **muddling up the different kind of neutrophils** (mature and immature).
        ''')

        with cont_4.expander('Chart'):
            st.image('./data/images/vgg19/vgg19_confuse.png')

    # ViT-b16
    if model_choice == "ViT-b16":

        # Preprocessing
        cont_vit = st.container()
        cont_vit.subheader("""Preprocessing""")

        cont_vit.markdown("""The data flow through this pipeline during training.
                        The same steps (except data augmentation) are applied to the test data.""")

        cont_vit.image('./data/images/vitb16/ViT_preprocessing.png')

        # Results
        cont_vit.subheader("""Model architecture""")

        cont_vit.markdown(
            """
            Vision transformers are composed of transformers encoder blocs that
            are fed linear projection of flattened images patches.

            For the ViT-b16, we have used our customised **MLP head** and
            **transfer learning** from ImageNet with a fine tuning of 5 transformer blocs.

            In addition to what was done for all models, we choose to improve generalisation
            by using **Rectified Adam** for optimiser
            and **label smoothing** on the softmax layer.

            Image were modified to the size **352 x 352** to fit the resolution
             required for patching in ViT-b16.


            """)
        common.display_md_gif(path='./data/images/vitb16/ViT.gif',
                              container=cont_vit,
                              alt_text='vit archi gif')

        cont_vit.subheader("""Main results""")

        cont_vit.markdown('''
                        - ###### Loss and global accuracy:  
        

        Global accuracy on training data: 90 % . 
        Global ** accuracy on test ** and validation data: **92 % **
        ''')

        with cont_vit.expander('Chart'):
            st.image('./data/images/vitb16/vitb16_training.png')

            st.markdown(''' There is no overfitting.
                        The validation loss is much lower than the training loss
                        because of the data augmentation and the class
                        weights used during the training and not validation.''')

        cont_vit.markdown('''
                        - ###### Classification report: 
         
        F1-Scores greater than 90% for all classes 
        **except for: PMY, MY, MMY and BNE**.
        ''')

        with cont_vit.expander('Table'):
            st.image('./data/images/vitb16/vitb16_report.png')

        cont_vit.markdown('''
                        - ###### Confusion matrix:  
        
        The model is **muddling up the different kind of neutrophils** (mature and immature).
        ''')

        with cont_vit.expander('Chart'):
            st.image('./data/images/vitb16/vitb16_confusion.png')
