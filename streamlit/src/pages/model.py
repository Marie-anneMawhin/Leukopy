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
                    fig, sorted_classes, sorted_preds = vgg19_utils.vgg19_prediction(model, img_file)
                    
                    st.text('P(%s) = %s'%(sorted_classes[0], vgg19_utils.print_proba(sorted_preds[0])))
                    st.text('P(%s) = %s'%(sorted_classes[1], vgg19_utils.print_proba(sorted_preds[1])))
                    st.text('P(%s) = %s'%(sorted_classes[2], vgg19_utils.print_proba(sorted_preds[2])))

                    #st.table({sorted_classes[i]:vgg19_utils.print_proba(sorted_preds[i]) for i in sorted_preds[:3]})
                    
                    st.subheader('Grad-CAM for %s:'%(sorted_classes[0]))
                    st.pyplot(fig)

                # if normalize_case2:
                #     pass
                # if normalize_case3:
                #     pass
