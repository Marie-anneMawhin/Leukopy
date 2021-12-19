import streamlit as st


def write():
    st.header('Perspectives')

    st.markdown("""Our models do well in classifying 10 classes of blood cell 
                pictures from 3 different datasets, 
                but **many improvements are possible**. Moreover, 
                we have seen that **good global metrics can hide important problems**, e.g. with 
                the model ability to **generalise** and correctly predict the nature of a previously unseen cell.""")

    cola, colb = st.columns([1, 12])
    with cola:
        st.image('./data/images/diversity.png', width=40)
    with colb:
        st.subheader('Increase the diversity')

    st.markdown("""Despite having merged 3 different datasets to increase generalisability, 
        our biggest problem remains the **lack of diversity for some classes**. Some solutions could involved: """)

    col1, col2 = st.columns([1, 3])

    with col2:
        # Lack of data and consequences

        st.markdown("""- more **diverse sources or types of pictures**:   
                i.e. coming from other institutions (different acquisition systems, different staining, luminosity etc...)
                will **allow for a more balanced dataset** (less SNE, and LY, **more immature**) and a **better ability for 
                the model to generalise on new data**""")

        st.markdown(""" - **then, we could develop data augmentation** :  
                for example we could try to mimic real-life histology staining, 
                alter actual staining or use **GANs** (Generative Adversarial Networks) 
                to produce new pictures""")

        st.markdown(""" - finally, we could consider **a complete training of our models** on 
                    blood cells pictures (and therefore discard ImageNet weights)""")

        st.markdown("""</br>
                </br>""", unsafe_allow_html=True)

    with col1:
        st.image('./data/images/augmentation.png')

    # Data labelling cross-validation
    colc, cold = st.columns([1, 12])
    with colc:
        st.image('./data/images/label.png', width=40)
    with cold:
        st.subheader('Improve the labelling')
    st.markdown(""" 
    We **need** to train the model with pictures that are **confidently labelled**.  
    One option is a **cross-validation process** between independent expert pathologists. 
    This was done in the Raabin project and revealed important disagreements in the labelling of neutrophils (BNE vs SNE, precursors).  
    This kind of work is time-consuming and resource-intensive, so **self-supervised 
    or semi-supervised learning** could be more reasonable alternatives.  
    
    Another option would be the use of **transcription factors and granule proteins as labels** . 
    This could be done by co-staining with antibodies or using flow cytometry labelling prior to staining.
    """)

    col3, col4 = st.columns([1, 6])
    with col3:
        ''
    with col4:
        st.image('./data/images/granulopoiesis.jpeg', width=500)

    st.markdown('''In the broader context of **cancer detection**, we could improve the ability of 
                        the model to **recognise immature cells** : PMY and MMY for example, 
                        but also **myeloblasts** (absent of our models because of lack of data), 
                        or lymphocytes precursors.  
                        Also it could be intersting to consider subclasses of lymphocytes (LY) such as: B, T and NK...''')

    # Object detection

    cole, colf = st.columns([1, 12])
    with cole:
        st.image('./data/images/detection.png', width=40)
    with colf:
        st.subheader('Object detection')

    col5, col6 = st.columns(2)
    with col5:
        st.image('./data/images/Im016_1.jpg', width=300)

    with col6:
        st.image('./data/images/bbox.png')

    st.markdown(""" We have only worked **on one step of the full process** : 
                we use segmented pictures, with only one cell at the centre. 
                **A possible extension of this work could involve 
                object (= blood cell) detection** (e.g. with YoloV5) on a large scale picture of 
                a blood smear, in order to produce ourselves the kind of pictures we use 
                in this app;""")
