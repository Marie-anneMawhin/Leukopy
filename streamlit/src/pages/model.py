import streamlit as st


model_list = ["vgg16", "vgg19", "ViT"]
print(f"Name: {3}\n Size:{3}\nType:{3}" )

def write():
    st.subheader('Classification')

    st.selectbox("Select model", options=model_list)

    
        
    img_file=st.file_uploader("Upload image",type=['jpg'])

    if img_file:
        img = img_file.read()

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Original image")
                
            st.image(img, width=150)
            
            st.text(f"""Name: {img_file.name}
        Size_poo:{img_file.size}
        Type:{img_file.type}""")
                
        with col2:
            with st.expander("Classified as"):
                pass
                
                # if normalize_case1:    # c'est ici que chaqu'un peut mettre son modèle
                #     pass
                # if normalize_case2:
                #     pass
                # if normalize_case3:
                #     pass