import streamlit as st

from importlib import reload

import utils.models.vgg19_utils as vgg19_utils
reload(vgg19_utils)


model_list = ["VGG16", "VGG19", "ViT-b16"]


def write():
    st.subheader('Classification')

    model_choice = st.selectbox("Select model", options=model_list)

    img_file = st.file_uploader(
        "Upload an image for classification",
        type=['jpg', 'png', 'tiff'],
    )

    if img_file:
        img = img_file.read()

        file_details = f"""
        Name: {img_file.name}
        Size:{img_file.size}
        Type:{img_file.type}"""

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original image ...")

            st.image(img, width=150)
            st.caption(file_details)

        with col2:
            with st.container():
                st.subheader("... is probably :")

                if model_choice == "VGG19":
                    # Importe le modèle (en cache)
                    model = vgg19_utils.load_model()
                    # Prédiction + Grad-CAM
                    fig = vgg19_utils.vgg19_prediction(model, img_file)
                    st.pyplot(fig)

                # if normalize_case2:
                #     pass
                # if normalize_case3:
                #     pass
