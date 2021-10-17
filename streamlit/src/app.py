import streamlit as st
from importlib import reload

import pages.home
reload(pages.home)

import pages.EDA
reload(pages.EDA)

import pages.model
reload(pages.model)

import pages.perspective
reload(pages.perspective)

import pages.about
reload(pages.about)




PAGES = {
    "Home": pages.home,
    "EDA": pages.EDA,
    "Modélisation": pages.model,
    "Perspectives": pages.perspective,
    "About": pages.about,
}


def main():
    """Main function of the App"""

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