import streamlit as st
import pandas as pd
import numpy as np
import umap

import plotly.express as px
from utils import eda_utils as eda


def write():
    st.title('EDA')

    df = pd.read_csv('src/data/df/PBC_dataset_normal_df_merged.csv')

    with st.expander('show dataframe'):

        st.dataframe(df.drop('label_2', axis=1).head())

    size_c = st.container()
    size_c.markdown('''
    # Image sizes
    The following interactive plots shows the resolution of images by dataset of origin.
    ''')
    sizes = df.groupby(['height', 'width', 'origin']).agg(
        count=('img_path', 'count'),
        log_count=('img_path', lambda x: np.log2(x.count()))).reset_index()

    fig_sizes = px.scatter(
        sizes, x='height', y='width', size='log_count',
        opacity=0.6, color='origin', hover_name='origin',
        hover_data={'origin': False,
                    'height': True,
                    'width': True,
                    'log_count': False,
                    'count': True
                    },
        template='seaborn')

    size_c.plotly_chart(fig_sizes)

    class_distr = st.container()
    class_distr.markdown(
        '''
    # Class distribution

    The following interactive plot shows the count of cells per classes.
    ''')

    fig_distrib = px.histogram(df, x='label',
                               template='none',
                               color='origin',
                               color_discrete_sequence=[
                                   '#4C78A8',  '#E45756', '#72B7B2'],
                               )
    class_distr.plotly_chart(fig_distrib)

    class_distr.markdown(
        '''
    We can see on this plot that the segmented neutrophils (SNE) are
    overrepresented in the Munich dataset. The lymphocyte population in majority
    in the Raabin and the Munich dataset. The classes are more balanced in the
    Barcelona dataset. Platelets are only present in the later dataset.
    ''')

    lumi = st.container()
    lumi.markdown('# Luminosity')

    lumi.image('src/data/images/plot_ramdom_cells.jpg',
               caption='Plot showing random cells for each label and dataset of origin.')

    lumi.markdown(
        '''
    We can see on this image that **each dataset present different coloration**. Moreover, certain
    types of cells look fairly similar: for example, monocytes
    and immature granulocytes or segmented and banded neutrophils. Some element could
    potentially interfer with the classification such as the **number of
    red blood cells** in the background or the image color.
    ''')

    umap_c = st.container()
