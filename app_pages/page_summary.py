import streamlit as st
import os
from PIL import Image


def page_summary_body():
    """Displays the project summary."""

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"This visual criteria are used to detect mildew on cherry leaves:\n\n"
        f"* **Cherry Leaf Spots:** Shows as purple spots turning brown or gray, "
        f"with possible white fungus in the center on top of leaves. \n\n"
        f"* **Powdery Mildew Cherry Leaves:** Shows as white patches on the underside of leaves, "
        f"which may become distorted.\n"
        f"Look for these signs on young leaves and in the inner canopy.\n\n"
        f"**Project Dataset**\n"
        f"* The available dataset contains of in total 4208 images\n with a 50 percent split of infected and uninfected cherry leaves."
    )
    st.write("---")
    st.write(
        f"* For additional information, please **visit** and **read** the "
        f"Project README.md file found here: \n\n[Cherry Leave Mold Detection Analysis](https://github.com/OV00VO/mildewdetectionincherryleaves/blob/main/README.md)\n"
        f"* For additional information about Cherry Leaf Mildew, please **visit** and **read** this on Wikipedia: \n\n"
        f"[Cherry leaf spot](https://en.wikipedia.org/wiki/Cherry_leaf_spot)"
    )
    st.write("---")
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate "
        f"infected with mildew and uninfected with mildew, in a visually way.\n"
        f"* 2 - The client is also interested in telling whether a given leaf contains mildew on the cherry leaf or not. "
    )
    
    image_dir = os.path.join(os.getcwd(), "images_project")
    col1, col2 = st.columns(2) 
    with col1:
        st.image(f"{image_dir}/powdery_mildew.JPG", caption="Infected Leaf: Cherry Leaf - Powdery Mildew")
    with col2:
        st.image(f"{image_dir}/healthy.JPG", caption="Uninfected Leaf: Cherry Leaf - Healthy")
    
    st.info(    
        f"* Cherry leaf images reveal distinct signs of powdery mildew, a fungal disease that could impact a cherry plantage cherry crops.\n\n"
        f"* Manually inspecting vast orchards is inefficient. To combat this, an ML system is proposed to analyze leaf images for instant mildew detection.\n\n" 
        f"* This would save time, scale better for numerous trees across farms, and potentially be applicable to other crops facing similar challenges.\n\n" 
        f"* By adopting machine learning, the company can streamline cherry production and potentially expand this technology for broader agricultural benefits.\n\n"
        f"As mentioned, spotting powdery mildew, that is a nasty fungus harming cherry crops, is crucial for farmers. But checking huge orchards by hand takes forever! "
        f"This project tackles that, by building a system that analyzes cherry leaf pictures to instantly detect mildew.\n\n" 
        f" The system learns from a massive dataset of 4208 images, half healthy and half infected." 
        f"This saves farmers tons of time and lets them treat problems faster, leading to healthier crops. " 
        f"Plus, this approach could potentially be used for other crops facing similar diseases in the future!"
        
    )
      
    st.success(
        f"Summary in bullet points:\n"
        f"* The analysis describes how to distinguish between healthy and mildew-infected leaves based on visual cues like spots and discoloration.\n\n"
        f"* Furthermore, it mentions the project dataset, containing a balanced number of images **4208** for both infected (powdery mildew) and healthy cherry leaves in an even split.\n\n"
        f"* Morever, provides links to learn more about the project itself and cherry leaf spot in general.\n\n"
        f"* Additionally, by using a visual tool it is possible to differentiate between healthy and mildewed leaves, "
        f"and they want the ability to determine if a specific leaf has cherry leaf mildew.\n\n"
    )
  