from pathlib import Path
from typing import Callable

import streamlit as st

import pandas as pd
import tensorflow as tf
from vit_keras import visualize
import matplotlib.pyplot as plt
from utils.common import *


def ViT_prediction(trained_model: tf.keras.Model, img_file: bytes,
                   dim: tuple = (320, 320),
                   label_map: dict = label_map,
                   preprocess: bool = False,
                   func_preprocessing: Callable = None,
                   ):

    array, img = get_img_array(
        img_file, dim, preprocess=preprocess, func_preprocessing=func_preprocessing)

    img = tf.keras.preprocessing.image.img_to_array(img)
    attention_map = visualize.attention_map(trained_model, img)

    probas = trained_model.predict(array)[0] * 100

    df_pred = pd.DataFrame({'label': label_map.keys(),
                            'pred_index': label_map.values(),
                            'proba': probas,
                            }).sort_values('proba', ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(attention_map)

    return fig, df_pred


def print_proba(df_pred: pd.DataFrame):

    st.text(f'P({df_pred.label[0]}) = {df_pred.proba[0]: .2f} %')
    st.text(f'P({df_pred.label[1]}) = {df_pred.proba[1]: .2f} %')
    st.text(f'P({df_pred.label[2]}) = {df_pred.proba[2]: .2f} %')
