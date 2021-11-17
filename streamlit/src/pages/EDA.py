import streamlit as st
import pandas as pd
import plotly.express as px
from utils import eda



def write():
    st.title('EDA')

    df = pd.read_csv('src/data/df/PBC_dataset_normal_df_merged.csv')
    
    # fig_cell = eda.plot_random_by_classes(df)
    # st.pyplot(fig_cell)

    with st.expander('show dataframe'):
        st.dataframe(df.head())

    with st.container():
        st.markdown(
        '''
        # Classe distribution

        The following interactive plot shows the count of cells per classes.
        ''')

        fig_distrib = px.histogram(df, x='label',
        template='none',
        color='origin',
        color_discrete_sequence=['#4C78A8',  '#E45756', '#72B7B2'],
        )
        st.plotly_chart(fig_distrib)

        st.markdown(
        '''
        We can see on this plot that the segmented neutrophils (SNE) are 
        overrepresented in the Munich dataset. The lymphocyte population in majority
        in the Raabin and the Munich dataset. The classes are more balanced in the 
        Barcelona dataset. Platelets are only present in the later dataset.


        # Luminosity
        ''')
        
        st.image('src/data/images/plot_ramdom_cells.jpg',
        caption='Plot showing random cells for each label and dataset of origin.')

        st.markdown(
        '''
        We can see on this image that each dataset present different coloration. Moreover, certain
        types of cells look fairly similar: for example, monocytes 
        and immature granulocytes or segmented and banded neutrophils. Some element could
        potentially interfer with the classification such as the number of 
        red blood cells in the background or the image color. 
        ''')

    sizes = df.groupby(['height', 'width']).agg(count=('img_path', 'count')).reset_index()
    st.dataframe(sizes)
    fig_sizes = px.scatter(
        sizes, x='height', y='width', size='count', 
        opacity=0.6, #range_x=(300,610), #ylim=(300,610), 
                     template='seaborn', )
    st.plotly_chart(fig_sizes)
                    #  height=500, width=600).options(line_alpha=0.7,
                    #                                 line_width=2,
                    #                                 fill_alpha=0.04)

import plotly.express as px

print(px.colors.qualitative.T10)