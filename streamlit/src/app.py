import pages.about
import pages.perspective
import pages.model
import pages.modelisation
import pages.analysis
import pages.EDA
import streamlit as st
from importlib import reload

import pages.home
reload(pages.home)
reload(pages.EDA)
reload(pages.modelisation)
reload(pages.model)
reload(pages.analysis)
reload(pages.perspective)
reload(pages.about)


PAGES = {
    "Home": pages.home,
    "EDA": pages.EDA,
    "Modelisation": pages.modelisation,
    "Analyse": pages.analysis,
    "Prediction": pages.model,
    "Perspectives": pages.perspective,
    "About": pages.about,
}


def main():
    """Main function of the App"""

    st.set_page_config(
        page_title="Leukopy",
        page_icon="🩸")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.write()
    # with st.spinner(f"Loading {selection} ..."):
    #     ast.shared.components.write_page(page)
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by the leukopy team. You can learn more about me at
        [leukopy](https://github.com/DataScientest/Leukopy).
"""
    )


if __name__ == "__main__":
    main()
