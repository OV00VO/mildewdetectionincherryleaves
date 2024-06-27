
# Mildew Detection in Cherry Leaves
This repository tackles cherry leaf mildew detection using a combination of data analysis and machine learning.

## Project Goals
* Develop data visualizations to differentiate healthy and mildewed cherry leaves.
* Build a machine learning model to predict mildew presence in cherry leaf images.

### Deployed APP on Heroku

[Mildew Detection in Cherry Leaves](https://mildewdetectionincherryleaves-c24b4ea4a636.herokuapp.com/)

## How to Use This Repo
1. Fork this repository on GitHub.
2. Copy the HTTPS URL of your forked repository.
3. Set up your cloud IDE workspace:
4. Log in and create a new workspace.
5. Paste the copied URL and click "Create."

## Access Jupyter Notebooks:
* Open the "jupyter_notebooks" directory.
* Select and run the desired notebook.

## Run the Project:
Use streamlit run app.py to launch the Streamlit dashboard.
* Streamlit App Dashboard Design

## The dashboard addresses the client's needs:

### Page 1: Quick Project Summary
* Introduceas the project and target audience (e.g., cherry farmers).
* Summarizes cherry leaf mildew and its challenges.
* Describes the dataset (size, source, image types).
* Outlines business requirements (visual differentiation and image-based prediction).
* Includes links for further learning about cherry leaf mildew.

### About the project:
Marianne McGuinness, the head of IT and Innovation at Farmy & Foods, an agricultural company, is facing a challenge. 
Their cherry plantations are affected by powdery mildew, a fungal disease harming various plants.
Cherry crops are among their finest products, and the company is concerned about compromised quality. 
Currently, employees manually inspect trees, spending 30 minutes per tree, taking leaf samples to visually identify mildew. 
If mildew is present, they apply a specific compound (taking 1 minute). 
With thousands of trees across multiple farms, this manual process is not scalable due to the time it takes. 
To save time, the IT team proposes an ML system that instantly detects mildew in cherry leaf images. 
A similar manual process exists for other crops. If successful, this project can be replicated for other crops. 
The dataset consists of cherry leaf images provided by Farmy & Foods, taken from their crops.

### User Stories: Build a dashboard to detect mildew in cherry leaves based on the following business requirements:
* The client wants a visual study to differentiate healthy and mildewed cherry leaves.
* The client wants to predict if a cherry leaf has mildew.

### Page 2: Project Hypothesis and Validation
* Explains the hypothesis for each business requirement.
* Details the validation methods used.
* Summarizes the validation results.

### Page 3: Mildew Visualizer (for Business Requirement 1)
* Provides checkboxes for visualizations:
* Differences between average and standard deviation images (infected vs. uninfected).
* Comparisons of average infected and uninfected leaf images.
* Image montages for infected or uninfected leaves.

### Page 4: Mildew Detector (for Business Requirement 2)
Uploader for multiple cherry leaf images.

Displays uploaded images with:
* Prediction statements (presence/absence of mildew) and probabilities.
* Summarizes results in a downloadable table.

### Page 5: ML Performance Metrics
* Shows label frequencies (infected/uninfected) across datasets (training, validation, test).
* Visualizes the model's learning process (accuracy and loss over training epochs).
* Summarizes model performance on the test set using metrics like accuracy, precision, recall, and F1-score.

Overall, this design ensures a user-friendly and informative dashboard that meets the client's requirements and facilitates data exploration, model evaluation, and actionable insights generation.

### Testing
**Comprehensive Testing Ensures Accuracy:** To guarantee the reliability and functionality of the cherry leaf mildew detection project, comprehensive testing was undertaken.  Not only were all features rigorously evaluated within the Streamlit app environment, mirroring the intended user experience, but each step of the code was also manually tested in a Jupyter Notebook to validate its logic and performance. **Additionally, a crucial aspect of the testing process involved manually testing a diverse set of cherry leaf images**, ensuring the machine learning functions accurately classified both healthy and mildew-infected leaves.

**Deployment Considerations:** While initial deployment attempts were made on Heroku, challenges arose due to the platform's soft limit of 300MB and hard limit of 500MB. The combined size of necessary packages, including Streamlit and TensorFlow-CPU, alongside project files, exceeded this limit. While Heroku support allowed a temporary increase to 600MB, it's important to note that this approach is not recommended for long-term use due to potential performance impacts. As a result, for similar machine learning projects with larger dependencies, deploying on alternative platforms designed for these needs is recommended.

### Future Enhancements:
**Automatic Photo Uploader:** Streamline the process by integrating an automatic photo uploader that allows users to directly capture and analyze cherry leaf images. This could be a mobile app or a web-based interface.

**Advanced Machine Learning:** Explore deeper neural networks or ensemble learning techniques to potentially improve detection accuracy and robustness.

**Severity Classification:** Extend the project beyond simple detection to identify the severity of mildew infection, providing valuable insights for treatment decisions.

**Multi-Disease Detection:** Enhance the model's capabilities to identify other common cherry leaf diseases, offering a broader diagnostic tool for farmers and researchers.

**Scalable Deployment:** Consider deploying the model as a RESTful API for integration with agricultural monitoring systems or other applications requiring mildew detection functionalities.

By incorporating these features, the project can evolve into a powerful and versatile tool for managing cherry leaf health.


### Credits
Code Institute: All code used has been remodeled, restructured, or recreated specifically for this project. Foundational concepts and learning materials were sourced from Code Institute's Machine Learning course and it is structured according to the learning objective/goals. To be able to develop this project, while learning some basic code structures was left for its functionality of the app.  

