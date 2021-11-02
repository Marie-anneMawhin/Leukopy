import streamlit as st

from importlib import reload

import data.model.vgg19.utils.vgg19_utils as vgg19_utils
reload(vgg19_utils)














model_list = ["vgg16", "vgg19", "ViT"]

def write():
    st.subheader('Classification')

    model_choice = st.selectbox("Select model", options=model_list)

        
    img_file=st.file_uploader(
        "Upload an image for classification",
        type=['jpg', 'png', 'tiff'],
        )

    if img_file:
        img = img_file.read()

        file_details= f"""
        Name: {img_file.name}
        Size:{img_file.size}
        Type:{img_file.type}"""
        

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Original image")
                
            st.image(img, width=150)
            st.caption(file_details)
                
        with col2:
            with st.expander("Classified as"):
                
                if model_choice == "vgg19":   
                    fig, gcams = vgg19_utils.load_vgg19(img_file)
                    st.pyplot(fig)
                    
                    #st.image([gcams[0], gcams[1], gcams[2]], width = 150)
                    
                # if normalize_case2:
                #     pass
                # if normalize_case3:
                #     pass