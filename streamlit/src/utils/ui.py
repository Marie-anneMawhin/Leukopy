from pathlib import Path
import streamlit as st

fname_dict = {'SNE': 'segmented neutrophil',
              'EO': 'eosinophil',
              'BA': 'basophil',
              'LY': 'lymphocyte',
              'MO': 'monocyte',
              'PLT': 'platelet',
              'ERB': 'erythroblast',
              'MMY': 'metamyelocyte',
              'MY': 'myelocyte',
              'PMY': 'promyelocyte',
              'BNE': 'band neutrophil'}

path_img = Path('./data/images/app_dataset')


def select_examples():

    folders_img = [f'{i.stem} - {fname_dict[i.stem]}'
                   for i in path_img.iterdir()]

    class_img = st.selectbox('a. Select a cell type', options=folders_img)

    class_path = path_img / class_img.split(' -')[0]

    dict_img = {image_path.stem: image_path
                for image_path in class_path.iterdir()}

    example_path = st.selectbox(
        'b. Select an example image', options=dict_img.keys())

    return dict_img, example_path


def upload_example():
    img_file = st.file_uploader(
        "Upload an image for classification",
        type=['jpg', 'png', 'tiff'],
    )

    return img_file
