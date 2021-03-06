from numpy.lib.shape_base import column_stack
import streamlit as st

from importlib import reload
from pathlib import Path
from PIL import Image

import utils.models.vgg19_utils as vgg19_utils
import utils.models.vgg16_utils as vgg16_utils
import utils.models.vit_b16_utils as vit_utils
from utils import common, ui

reload(vgg19_utils)
reload(vgg16_utils)
reload(vit_utils)
reload(common)

model_list = ["VGG16+SVM", "VGG19", "ViT-b16"]


def write():
    st.header('Prediction')

    st.write('')
    st.subheader('1. Choose which model you want to use for prediction')

    model_choice = st.selectbox("Select a model", options=model_list)

    st.write('')
    st.subheader('2. Upload an image or select a preloaded example')
    st.markdown(
        '*Note: please remove any uploaded image to choose an example image from the list.*')
    cola, colb, colc = st.columns([4, 1, 4])

    with cola:
        uploaded_img = ui.upload_example()

    with colb:
        st.markdown(
            '''<h3 style='text-align: center; '> 
            <br>
            <br>
            OR
            </h3>''', unsafe_allow_html=True)

    with colc:
        dict_img, example_path = ui.select_examples()

    if uploaded_img:
        img_file = uploaded_img
        img_name = img_file.name

    elif example_path:
        selected_img = dict_img[example_path]
        img_file = open(selected_img, 'rb')

        try:
            img_name = img_file.name.split('/')[-2]

        except IndexError:
            st.write('please load an image.')

    img = img_file.read()

    img_info = Image.open(img_file)
    file_details = f"""
    Name: {img_name}
    Type: {img_info.format}
    Size: {img_info.size}
   
    """

    st.write('')
    st.subheader('3. Results')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image ...")

        st.image(img, width=150)
        st.caption(file_details)

    with col2:
        with st.container():
            st.subheader("... is probably :")

            if model_choice == "VGG19":
                # Importe le mod??le (en cache)
                model = vgg19_utils.load_model()
                # Pr??diction + Grad-CAM
                fig, sorted_classes, sorted_preds = vgg19_utils.vgg19_prediction(
                    model, img_file)

                st.text('P(%s) = %s' % (
                    sorted_classes[0], vgg19_utils.print_proba(sorted_preds[0])))
                st.text('P(%s) = %s' % (
                    sorted_classes[1], vgg19_utils.print_proba(sorted_preds[1])))
                st.text('P(%s) = %s' % (
                    sorted_classes[2], vgg19_utils.print_proba(sorted_preds[2])))

                st.subheader('Grad-CAM for %s:' % (sorted_classes[0]))
                st.pyplot(fig)

            if model_choice == "VGG16+SVM":
                # Choix du mod??le
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

                _, superimposed_img = vgg16_utils.gradcam(
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
