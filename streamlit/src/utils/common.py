from pathlib import Path
import base64
import streamlit as st
from streamlit import delta_generator as dt


def display_md_gif(path: Path, container: dt.DeltaGenerator, alt_text: str = None):
    file_ = open(path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    container.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="{alt_text}">', unsafe_allow_html=True)
