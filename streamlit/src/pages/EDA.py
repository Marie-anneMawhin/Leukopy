import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from traitlets.traitlets import default


def write():
    st.title('EDA')

    df = pd.read_csv('src/data/df/PBC_dataset_normal_df_merged.csv')

    with st.expander('show dataframe'):

        st.dataframe(df.drop('label_2', axis=1).head())

    size_c = st.container()
    size_c.markdown('''
    # Image sizes
    The following interactive scatter plot shows the resolution of images by dataset of origin.
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

    size_c.markdown('''
   The images of the Munich dataset shows a consistent image size. 
   The Barcelone dataset shows variations in size, and
   the Raabin dataset has several single images with different sizes.
    ''')

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
    Segmented neutrophils (SNE) are **overrepresented** in the Munich dataset. 
    The lymphocyte population in majority in the Raabin and the Munich dataset. 
    The classes are more **balanced** in the Barcelona dataset. 
    Platelets are only present in the later dataset.
    ''')

    lumi = st.container()
    lumi.markdown('# Luminosity')

    lumi.image('src/data/images/plot_ramdom_cells.jpg',
               caption='Plot showing random cells for each label and dataset of origin.')

    lumi.markdown(
        '''
    This image shows that each dataset present **different staining and exposure**. 
    Moreover, certain types of cells appear quite similar: for example, monocytes
    and immature granulocytes or segmented and banded neutrophils. Some element could
    potentially interfere with the classification such as the **number of
    red blood cells** in the background or the image color.
    ''')

    umap_c = st.container()

    umap_c.markdown(
        '''
    # Dimension Reduction: UMAP
    The following interactive plot shows a sample of the dataset after 
    reduction of dimension with Uniform Manifold Approximation and Projection.
    Platelets and erythroblats are isolated from the rest of the cell population.
    Moreover, the origin of the dataset explains partly how data are clustered.
    ''')

    df_umap = pd.read_csv('src/data/df/UMAP.csv')

    choice_umap = umap_c.multiselect('Select the dataset of origin',
                                     options=df_umap.origin.unique(),
                                     default=df_umap.origin.unique())

    embedding = df_umap[df_umap.origin.isin(choice_umap)]

    umap_plot = px.scatter_3d(
        embedding, x='UMAP_1', y='UMAP_2', z='UMAP_3',
        color='label', opacity=0.9,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template='plotly_dark')

    umap_plot.update_traces(marker=dict(size=2.5))

    umap_plot.update_layout(title='UMAP embedding',
                            legend=dict(itemsizing='constant',
                                        itemwidth=45,
                                        ),
                            margin=dict(l=0, r=0, b=0, t=80),
                            width=800,
                            height=800
                            )

    range_plot_x = [df_umap.UMAP_1.min(), df_umap.UMAP_1.max()]
    range_plot_y = [df_umap.UMAP_2.min(), df_umap.UMAP_2.max()]
    range_plot_z = [df_umap.UMAP_3.min(), df_umap.UMAP_3.max()]

    umap_plot.update_scenes(xaxis=dict(range=range_plot_x),
                            yaxis=dict(range=range_plot_y),
                            zaxis=dict(range=range_plot_z))

    umap_c.plotly_chart(umap_plot)
