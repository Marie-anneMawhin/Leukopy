import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def write():

    st.title('Analysis')

    st.markdown("""Here, we detail and discuss the results obtained with VGG19. 
                Most of the following remarks apply to the others models, as emphasized in the report.""")

    # Granulocytes
    cont_1 = st.container()
    cont_1.subheader("What does the confusion matrix reveal ?")

    cont_1.markdown(
        '''The **strongest percentages of misclassified pictures have been circled in red** on the matrix below.''')

    with cont_1.expander('Confusion matrix'):
        st.image('./data/images/vgg19/vgg19_confuse_red.png', width=500)

    cont_1.markdown('''We remark **some pathological cases we are going to discuss now** : 
                    the **neutrophilic granulocytes and their precursors**. To understand why,
                    we need to invoke biology :''')

    with cont_1.expander('The life of a neutrophilic granulocyte ...'):
        st.image('./data/images/vgg19/granulocytes.png')

    cont_1.markdown('''PMY, MY, MMY, BNE and SNE are steps in the neutrophilic granulocyte **maturation process**. 
                    This process is **a continuous one** (e.g. the nucleus slowly evolve from a potato shape to a multi-lobed one), 
                    so it is believable we find some cells with features of two successive growth steps.''')

    cont_1.markdown('''The model must choose a class, then we can get classification errors. 
                    **A possible solution : display the probabilities for the most probable classes !**''')

    cont_1.markdown(
        '''We could also **invert the viewpoint** and invoke possible **labelling errors** (pictures below) :''')

    with cont_1.expander('Some disturbing cases ...'):
        st.image('./data/images/vgg19/confusions.png',
                 caption='From left to right : PMY, two MMY and a BNE.')

        st.markdown('''The two left pictures share noticeable features (e.g. same nucleus "bean" shape, cytoplasmic granulations), 
                    and we could say the same for the two right pictures (C-shaped nucleus). The third picture could be a 
                    "young" BNE, or an "old" MMY because of nucleus shape.
                    Labeling such pictures is a complex work, and requires well trained
                    experts which are not unerring.''')

    cont_1.markdown('''These two reasons could explain why the model performs 
                    so bad **only** on neutrophilic granulocytes (especially MMY and MY).''')



    # G??n??ralisation
    cont_2 = st.container()
    cont_2.subheader("About the ability to generalise")

    with cont_2.expander('Click to unwrap :'):
        st.markdown("""The table below contains percentage of correctly classified pictures for each class and for each dataset 
                    (Barcelona, Munich, Raabin). We highlight in yellow classes for which we have Raabin data.""")

        # Accuracy per class and dataset
        st.image('./data/images/vgg19/well_classified_stats.png')

        # Populations
        #df = pd.read_csv('./data/df/PBC_dataset_normal_df_merged.csv')

        st.markdown('''
        The following plot shows the count of cells per classes.
        ''')
        #fig_distrib = px.histogram(df, x='label',
        #                           template='none',
        #                           color='origin',
        #                           color_discrete_sequence=['#4C78A8',  '#E45756', '#72B7B2'])
        #cont_2.plotly_chart(fig_distrib)

        st.markdown('''Data from Munich and Barcelona look good : bright and clear pictures, 
        but lack of variety. **Raabin data are more diverse, 
        which is good for our model ability to generalise** : we obtain good metrics when we 
        have Raabin data, even for BA (a minority class).''')

        st.markdown('''We **reveal a bias** when considering BNE (Barcelona dominant) and SNE (Munich dominant) : BNE from Munich and
        SNE from Barcelona obtain **a meager 60%**. The **source imbalance has a visible effect**, and **the lack of 
        Raabin data to enrich the model implies it may not be able to generalise well on these classes** despite global 
        **SNE F1-Score greater than 95%** and despite **SNE are the dominant class in most of the available datasets and in circulating blood**.
        ''')

        st.markdown(
        '''This emphasizes the **necessity to collect data from different institutions** in order to make diverse datasets.''')

        st.markdown('''Good performances obtained on BA, or ERB or MO implies 
        the required global amount of data for each class is not too high.
        **We need diversity, probably not high numbers**.
        ''')

    # Grad-CAM
    cont_3 = st.container()
    cont_3.subheader("Explainability")
    cont_3.markdown("""
    To assess how our models performed we employed the following techniques:
    """)

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
        where $W$ and $H$ respectively are the width and the height of all feature maps.
        ''')

    with cont_3.expander('Attention map'):
        st.markdown('''The attention map is the **average of attention weights accross all heads**. 
        The resulting weight matrices are **recursively multiply for all layers** of the ViT.
        ''')
        st.image('./data/images/attention.png')

        st.markdown(r''' 
        For each **single head**: 
        $$
        Attention (Q, K, V) =  softmax(\frac{QK^T}{\sqrt{d}})V
        $$
        where
        $Q = XW^Q$,  $K = XW^K$ and $V = XW^V$.''')

        st.text(" \n ")

        st.markdown(r''' 
        For multiple heads:
        $$
        MultiHead (Q,K,V)= Concat (head_1???,???, head_h???)W^O
        $$
        where $head_i= Attention (QW_i^Q,KW_i^K,VW_i^V)$
        $$''')

        st.text(" \n ")

        st.markdown(r'''
        For a transformer with $L$  layer, the attention map $??$ is calculated recursively as:
        $$
        ??(l_i) = A(l_i)A(l_{i-1})
        $$
        ''')
