from pathlib import Path
import base64
import streamlit as st
from streamlit import delta_generator as dt
from typing import Callable

import numpy as np
import tensorflow as tf
from PIL import Image


def display_md_gif(path: Path, container: dt.DeltaGenerator, alt_text: str = None):
    file_ = open(path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    container.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="{alt_text}">', unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_model(model_path: Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model


@st.cache()
def get_img_array(img_file: bytes, dim: tuple,
                  preprocess: bool = True,
                  func_preprocessing: Callable = None) -> tuple:
    img = Image.open(img_file)
    img = img.convert('RGB').resize(dim)
    array = np.array(img)
    array = np.expand_dims(array, axis=0)

    if preprocess == True:
        array = func_preprocessing(array)
    return array, img


################### dict ###################
classes = ["BA", "BNE", "EO", "ERB", "LY",
           "MMY", "MO", "MY", "PLT", "PMY", "SNE"]
label_map = {'BA': 0, 'BNE': 1, 'EO': 2, 'ERB': 3, 'LY': 4,
             'MMY': 5, 'MO': 6, 'MY': 7, 'PLT': 8, 'PMY': 9, 'SNE': 10}
