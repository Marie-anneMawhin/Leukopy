import streamlit as st

from importlib import reload
from pathlib import Path

import utils.models.vgg19_utils as vgg19_utils
import utils.models.vgg16_utils as vgg16_utils
import utils.models.vit_b16_utils as vit_utils
from utils import common

reload(vgg19_utils)
reload(vgg16_utils)
reload(vit_utils)
reload(common)

model_list = ["VGG16+SVM", "VGG19", "ViT-b16"]


def write():
    st.subheader('Choose which model you want to use for prediction :')

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
                    fig, sorted_classes, sorted_preds = vgg19_utils.vgg19_prediction(
                        model, img_file)

                    st.text('P(%s) = %s' % (
                        sorted_classes[0], vgg19_utils.print_proba(sorted_preds[0])))
                    st.text('P(%s) = %s' % (
                        sorted_classes[1], vgg19_utils.print_proba(sorted_preds[1])))
                    st.text('P(%s) = %s' % (
                        sorted_classes[2], vgg19_utils.print_proba(sorted_preds[2])))

                    # st.table({sorted_classes[i]:vgg19_utils.print_proba(sorted_preds[i]) for i in sorted_preds[:3]})

                    st.subheader('Grad-CAM for %s:' % (sorted_classes[0]))
                    st.pyplot(fig)

                if model_choice == "VGG16+SVM":
                    # Choix du modèle
                    model_flag = vgg16_utils.choose_model(img_file)

                    if model_flag == "VGG16_SVM_6_C_SF_flag":
                        base_model, str_result, img = vgg16_utils.VGG16_SVM_6_C_SF(
                            img_file)
                        st.write(str_result)

                    if model_flag == "VGG16_SVM_6_C_AF_flag":
                        base_model, str_result, img = vgg16_utils.VGG16_SVM_6_C_AF(
                            img_file)
                        st.write(str_result)

                    if model_flag == "VGG16_SVM_8_C_AF_flag":
                        base_model, str_result, img = vgg16_utils.VGG16_SVM_8_C_AF(
                            img_file)
                        st.write(str_result)

                    if model_flag == "VGG16_SVM_8_C_SF_flag":
                        base_model, str_result, img = vgg16_utils.VGG16_SVM_8_C_SF(
                            img_file)
                        st.write(str_result)

                        big_heatmap, superimposed_img = vgg16_utils.gradcam(
                            base_model, img, img_file, alpha=0.8, plot=False)
                    st.image(superimposed_img, width=150)

                if model_choice == "ViT-b16":

                    VIT_PATH = Path('./data/model/vitb16')
                    model = common.load_model(model_path=VIT_PATH)

                    fig, df_pred = vit_utils.ViT_prediction(model, img_file)

                    vit_utils.print_proba(df_pred)

                    st.subheader(
                        f'Attention map for {df_pred.label[0]}:')
                    st.pyplot(fig)
