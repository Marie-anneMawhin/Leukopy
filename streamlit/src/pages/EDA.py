import streamlit as st
import pandas as pd
import plotly.express as px
from utils import eda




def write():
    st.title('EDA')

    df = pd.read_csv('src/data/df/PBC_dataset_normal_df.csv')
    
    # fig_cell = eda.plot_random_by_classes(df)
    # st.pyplot(fig_cell)

    with st.expander('show dataframe'):
        st.dataframe(df.head())

    with st.expander('Cell distribution'):
        st.write('Here is the count of cells per classes')

        fig_distrib = px.histogram(df, x='label',
        template='none',
        color='label'
        )
        st.plotly_chart(fig_distrib)

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