import streamlit as st


def write():
    st.subheader('Perspectives')
    
    st.markdown("""Our models do well in classify ten classes of blood cells 
                pictures from three different datasets, 
                but **many improvements are possible**. Moreover, 
                we have seen that **good global metrics can hide important problems**, e.g. with 
                the model ability to generalise and correctly predict the nature of a previously unseen cell.""")

    # Lack of data and consequences
    st.markdown("""Our biggest problem is the **lack of diverse and accurately labelled data** : """)
                
    st.markdown("""- more **diverse and labelled pictures**, 
                i.e. coming from other institutions (different acquisition systems, different stainings, luminosity etc...)
                will **allow for a more balanced dataset** (less SNE, and LY, **more immatures**) and a **better ability for 
                the model to generalise on new data**;""")
    
    st.markdown(""" - **then, we could develop data augmentation** : 
                for example we could try to mimic real-life histology stainings, 
                alter actual stainings or use GANs (Generative Adversarial Networks) 
                to produce new pictures;""")
                
    st.markdown(""" - finally, we could consider **a complete training of
                our models** on blood cells pictures (and therefore discard ImageNet weights);""")
    
    st.markdown("""</br>
                </br>""", unsafe_allow_html=True)
                
    st.markdown("""In order to help with **cancer diagnosis**, we must improve the ability of 
                the model to **recognise immature cells** : PMY and MMY for example, 
                but also **myeloblasts** (absent of our model because of lack of data), or lymphocytes precursors (see below).
                About the lymphocytes, we consider them as a unique class, 
                whereas there are at least three : B, T and NK.""")
                
    st.image('./data/images/lymphocytes_growth.png')
       
    
    # Data labelling cross-validation
    st.markdown(""" We **need** to train the model with **pictures we 
                absolutely are certain of the label** : this requires a 
                **cross-validation process between independant and experimented 
                specialists**. For example, this kind of work has been done and 
                published by members of the Raabin project and the results 
                reveal important disagreements in the labelling of neutrophils 
                (BNE vs SNE, immatures). This kind of work is huge, so **self-supervised 
                or semi-supervised learning** are possibly more reasonable alternatives. """)    
    
    # Object detection
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(""" We have only worked **on one step of the full process** : 
                    we use segmented pictures, with only one cell at 
                    the center. 
                    **A possible extension of this work could involve 
                    object (= blood cell) detection** (e.g. with YoloV5) on a large scale picture of 
                    a blood smear, in order to produce ourselves the kind of pictures we use 
                    in this app;""")
    with col2:
        st.image('./data/images/Im016_1.jpg', width = 300)
    

