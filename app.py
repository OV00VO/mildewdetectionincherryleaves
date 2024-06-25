# Credits to Code Institute: All code below is either; 
# re-modeled, re-structed or re-created to fit this project.

import streamlit as st
from app_pages.multipage import multipage

from app_pages.page_summary import get_summary_content
from app_pages.page_mildew_visualizer import get_visualizer_content
from app_pages.page_mildew_detector import get_detector_content
from app_pages.page_project_hypothesis import get_hypothesis_content
from app_pages.page_ml_performance import get_performance_metrics

app = MultiPage(app_name="Mildew Detection in Cherry Leaves")

page_names = {
    "Project Overview": get_summary_content,
    "Visualize Mildew": get_visualizer_content,
    "Detect Mildew in Leaves": get_detector_content,
    "Project Hypothesis": get_hypothesis_content,
    "ML Model Performance": get_performance_metrics,
}


def display_current_page(page_names, chosen_function):
    current_page = [page for page, func in page_names.items() if func == chosen_function]
    if current_page:
        st.write(f"**Current Page:** {current_page[0]}")


for page_name, page_function in page_names.items():
    if st.sidebar.button(page_name):
        page_function()
        
        display_current_page(page_names, page_function)

app.run()