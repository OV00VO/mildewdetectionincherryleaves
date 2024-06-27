# Credits to Code Institute: All code below is either; 
# re-modeled, re-structed or re-created to fit this project.

import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* From the **Cherry Leaf Mildew Detection** and **Cherry Leaf Mildew Visualiser** in the **Menu**, the farmer suspect mildew on the cherry leaf, and it have clear marks/signs, "
        f"typically on small parts of the leaf at first, that can differentiate them from an uninfected leaf. \n\n"
        f"* An **Image Montage** shows that typically a mildew leaf has powdery mildew across it. "
        f"**Average Image**, **Variability Image** and **Difference between Averages studies**, did reveal "
        f"a clear pattern to differentiate one from another.\n"
    )
    
    st.info (
        f"While visual inspection can identify early signs of powdery mildew on cherry leaves as distinct marks on small areas, machine learning offers a powerful tool for large-scale analysis. " 
        f"Techniques like studying average and variability images of cherry leaves reveal patterns that differentiate infected from healthy leaves. \n\n" 
        f"This suggests ML or AI systems, trained on such patterns, could automate mildew detection, improving efficiency and scalability in cherry orchards."
    )
    
