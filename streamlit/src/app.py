import streamlit as st

page = st.sidebar.radio('Navigation', options = ['EDA', 'Modélisation'])

st.title('Leukopy - blood cell image classifier')


# if page == 'EDA':
#     st.title("Démo Streamlit Mar21 DA DS")