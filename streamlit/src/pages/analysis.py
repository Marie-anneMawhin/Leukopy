import streamlit as st

def write():
    
    st.title('Analysis')
    
    st.markdown("""Here, we detail and discuss the results obtained with VGG19. 
                Most of the following remarks apply to the others models, as emphasized in the report.""")
    
    ## Granulocytes
    cont_1 = st.container()
    cont_1.subheader("What does the confusion matrix reveal ?")
    
    with cont_1.expander('Charts'):
        st.image('./data/images/vgg19/vgg19_confuse.png')
    
    ## Généralisation
    cont_2 = st.container()
    cont_2.subheader("About the ability to generalise")
    
    with cont_2.expander('Charts'):
        # Discuter les métriques en fonction de l'origine des images : biais + importance de Raabin
        st.image('./data/images/vgg19/well_classified_stats.png')


    ## Grad-CAM
    cont_3 = st.container()
    cont_3.subheader("Grad-CAM")
    
    with cont_3.expander('Grad-CAM'):
        st.markdown(''' The Grad-CAM technique to generate CAMs (*class activation maps*) 
        helps us visualize what our CNN-based models look in the given data.
        ''')
        st.image('./data/images/gradcam_1.png')
         
        st.markdown(r''' 
        **Class activation map definition** : for a given class **(SNE)**, we compute the CAM 
        $$
        L^\text{SNE} = \text{ReLU(S)}
        $$
        where
        $$
        S = \sum_{k = 1}^{512} \alpha_k^\text{SNE} A^\text{<k>}
        $$
        and
        $$
        \alpha^\text{SNE}_k = \frac{1}{W \times H} \sum_i \sum_j \frac{\partial y^\text{SNE}}{\partial A_{ij}^{<k>}}
        $$
        where W and H respectively are the width and the height of all feature maps.
        ''')

