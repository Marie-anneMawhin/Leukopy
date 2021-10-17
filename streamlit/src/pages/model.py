import streamlit as st


model_list = ["vgg16", "vgg19", "ViT"]

def write():
    st.subheader('Classification')

    st.selectbox("Select model", options=model_list)

        
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
                pass
                
                # if normalize_case1:    # c'est ici que chaqu'un peut mettre son modèle
                #     pass
                # if normalize_case2:
                #     pass
                # if normalize_case3:
                #     pass