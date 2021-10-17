import streamlit as st

# page = st.sidebar.radio("Navigation", options = ["EDA", "Modélisation"])

st.title("Leukopy - blood cell image classifier")

import pages.home
import pages.EDA
import pages.model
import pages.perspective
import pages.about
# if page == "EDA":
#     st.title("Démo Streamlit Mar21 DA DS")

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