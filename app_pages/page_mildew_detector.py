import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities,
)

def page_mildew_detector_body():
    
    st.success(
       f"Stop wasting time on manual inspections!"
       f"This ML-powered tool analyzes cherry leaf images for instant mildew detection. "
       f"Upload a cherry leaf (or several!), get results instantly, and save precious time managing your orchard." 
    )
     
    st.info(
        f"* **User Story:** The client is interested in telling whether a given cherry leaf contains mildew on cherry leaves "
        f"or not.\n\n Here you can test the it fast functionality and download a **.csv** file with the results."
    )

    st.write(
        f"* Download a set of infected and uninfected cherry leaves for live prediction. "
        f"You can download the images from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        "Upload cherry leaf samples. You may select more than one.", type="jpg", accept_multiple_files=True
    )
    
    st.info(    
        f"Struggling to keep up with powdery mildew in your cherry orchard??\n\n " 
        f"Manually checking every leaf takes ages! This project uses machine learning to analyze cherry leaf images for instant mildew detection.\n\n"
        f"**Imagine this:** Upload a picture of a cherry leaf (or several!), and the system instantly tells you if there's a problem. " 
        f"No more endless manual inspections! - The benefits?\n\n"
    )
    
    st.success(    
        f"**Save time:** Ditch the tedious hand-checking and get results instantly.\n\n"
        f"**Scale up:** Analyze leaves from hundreds of trees across your farm with ease.\n\n"
        f"**Future potential:** This technology might even help with other crops facing similar issues!\n\n"
        )
    
    st.info(    
        f"By embracing machine learning, you can streamline cherry production, "
        f"potentially expand this tech to other crops, and contribute to a more productive and sustainable agricultural future.\n\n"
        f"**Below you can see the result of the analysis, and remember to use the above recommended dataset of cherry leaves.**"
        )
     
    if images_buffer is not None:
        df_report = pd.DataFrame(columns=["Name", "Result"])

        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Mildew Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = "v1"
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            print(f"Appending data: Name - {image.name}, Result - {pred_class}")

            new_row = pd.DataFrame({"Name": [image.name], "Result": [pred_class]})
            df_report = pd.concat([df_report, new_row], ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)

            st.download_button(
                "Download Report",
                df_report.to_csv(index=False),
                file_name="report.csv",
            )


if __name__ == "__main__":
    page_mildew_detector_body()