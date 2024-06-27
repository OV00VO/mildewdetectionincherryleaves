# Credits to Code Institute: All code below is either; 
# re-modeled, re-structed or re-created to fit this project.

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### This is how the Frequencies Differed Across the Train, Validation, and Test Sets.")
    
    st.write(
        f"* For additional information about ML datasets, please **visit** and **read** this on Wikipedia: \n\n"
        f"[List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)"
        )

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    
    st.success(
    f"In this diagram you can se how the train, validation and test set of healthy and powdery mildew cherry leaves performed in the ML learning process. " 
    f"As seen in the Accuracy this normally should strive towards High and the Loss should be Low, atleast at the end of the ML learning process."   
    )
    
    st.write("---")
  

    st.write("### A Model's Evolution: From Image Data to Deployment of ML Model")
    col1, col2 = st.columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.jpg")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.jpg")
        st.image(model_loss, caption='Model Training Losses')
    
    st.success(
    f"This shows how the Accuracy and Loss was performing, when analysing cherry leaves images, in the Machine Learning process."  
    )
    st.write("---")

    st.write("### Testing True Performance, and Gap of the ML model")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    st.success(
    f"To sum this up, the ML learning process has a loss that strives to Zero and the Accuracy strives to be close to One."  
    )
    st.write("---")