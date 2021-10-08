# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:16:17 2021

@author: laleh
"""

import streamlit as st
import pandas as pd
import numpy as np




def main():
    st.title("Classification  App")
    
    menu=["Classification", "Filtering"]
    choice=st.sidebar.selectbox("Menu",menu)
    
    if choice=="Classification":
        st.subheader("Classification")
        image_file=st.file_uploader("Upload image",type=['jpg'])
        
        normalize_case1=st.sidebar.checkbox("Model1")
        normalize_case2=st.sidebar.checkbox("Model2")
        normalize_case3=st.sidebar.checkbox("Model3")
        
        if image_file is not None:
           
            image=image_file.read()   
            
            file_details={"Filename":image_file.name,"File shape":image_file.size,"Filetype":image_file.type}           
            st.write(file_details)
            
            
            
            
            col1,col2=st.beta_columns(2)
            with col1:
                with st.beta_expander("Original Image"):
                    
                  st.image(image,width=150)
                    
            with col2:
                with st.beta_expander("Classified as"):
                    
                    if normalize_case1:    # c'est ici que chaqu'un peut mettre son modèle
                        pass
                    if normalize_case2:
                        pass
                    if normalize_case3:
                        pass
                   
    else:
        st.subheader("Filtering")
    
    
    
if __name__=='__main__':
    main()