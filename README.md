
# Mildew Detection in Cherry Leaves: A Business Case for Streamlined Farming

[Mildew Detection in Cherry Leaves - Deployed on Heroku](https://mildewdetectionincherryleaves-c24b4ea4a636.herokuapp.com/)

[GitHub Repository for this Project](https://github.com/OV00VO/mildewdetectionincherryleaves.git)

---
### About the Project: The Story Behind the Business Case
Marianne McGuinness, the head of IT and Innovation at Farmy & Foods, an agricultural company, is facing a challenge. Their cherry plantations are affected by powdery mildew, a fungal disease harming various plants. Cherry crops are among their finest products, and the company is concerned about compromised quality. Currently, employees manually inspect trees, spending 30 minutes per tree, taking leaf samples to visually identify mildew. If mildew is present, they apply a specific compound (taking 1 minute). With thousands of trees across multiple farms, this manual process is not scalable due to the time it takes. To save time, the IT team proposes an ML system that instantly detects mildew in cherry leaf images. A similar manual process exists for other crops. If successful, this project can be replicated for other crops. The dataset consists of cherry leaf images provided by Farmy & Foods, taken from their crops.
## Challenge
### Tackling Cherry Leaf Mildew with Machine Learning
Cherry leaf mildew, a fungal disease, poses a significant threat to cherry orchards, potentially reducing crop quality. Traditionally, farmers rely on manual inspection of vast orchards, a time-consuming and error-prone process. This inefficiency is particularly felt by companies like Farmy & Foods, which manage thousands of trees across multiple farms, spending 30 minutes per tree on visual inspection through leaf samples. This project tackles this critical challenge by presenting a solution that utilizes image analysis and machine learning (ML) for automated cherry leaf mildew detection. A user-friendly Streamlit dashboard serves as the interface for visualization and prediction, empowering cherry farmers with efficient crop management tools. This ML-powered approach offers significant benefits compared to traditional manual inspection, paving the way for improved farm efficiency and broader impact within the agricultural domain.
### User Stories: Build a dashboard to detect mildew in cherry leaves based on the following business requirements
* The client wants a visual study to differentiate healthy and mildewed cherry leaves.
* The client wants to predict if a cherry leaf has mildew.

### Here's how this project translates into a compelling business case for cherry farmers
**Problem:** Manually inspecting vast cherry orchards for mildew is time-consuming, inefficient, and prone to human error.\
**Solution:**
This project proposes an automated cherry leaf mildew detection system using machine learning (ML) and image analysis. By leveraging image recognition, the system can instantly analyze cherry leaf images and identify the presence of mildew.
* **Automated Detection:** The project presents an ML-powered system that analyzes cherry leaf images for instant mildew detection. This eliminates the need for tedious manual checks, saving cherry farmers valuable time and resources.
* **User-Friendly Interface:** A Streamlit dashboard provides a user-friendly interface for visualization and prediction. Farmers can easily upload cherry leaf images and receive instant results, helping them make informed decisions about crop management.
### Benefits
* **Increased Efficiency:** Automated mildew detection frees up farmers' time to focus on other critical tasks, boosting overall farm productivity.
* **Cost-Effective Solution:** Compared to manual inspection, the proposed ML system offers a scalable and cost-effective approach to disease detection.
* **Improved Crop Yield:** Early identification of mildew allows for timely treatment, potentially minimizing crop damage and increasing yield.
### Enhancing Efficiency and Profitability for Cherry Farmers
An ML-powered system specifically designed for automated mildew detection in cherry leaves offers significant advantages to cherry farmers. By addressing the limitations of manual inspection, the project delivers a cost-effective and scalable solution compared to traditional methods. This translates to improved farm efficiency, potentially leading to increased crop yield and overall farm profitability.
### The ML Advantage
A machine learning model, specifically a Convolutional Neural Network (CNN), will be trained to analyze cherry leaf images and predict the presence of mildew. This model offers several advantages:
* **Accuracy:** CNNs excel at image recognition tasks, achieving high accuracy in identifying mildew based on image features like color, texture, and leaf discoloration.
* **Scalability:** The ML model can efficiently analyze large volumes of cherry leaf images, making it suitable for farms of all sizes.
* **Objectivity:** Unlike manual inspection, the ML model is objective, eliminating the possibility of human error in identifying mildew.
### Scalability and Potential for Broader Impact
The project doesn't stop at cherry trees. Further exploration can investigate the application of this technology to other crops facing similar disease challenges. By framing the project within this business case structure, we demonstrate the value proposition of the ML solution and its potential to address challenges faced by a wider range of farmers.
### Deployment and Future Potential
The project aims for deployment on a platform suitable for machine learning applications. Additionally, the core technology has the potential to be adapted for disease detection in other crops facing similar challenges, expanding its impact on the agricultural domain. Further enhancements could include:
* **Automatic Photo Uploader:** Streamlining the process by allowing direct capture and analysis of cherry leaf images through a mobile app or web interface.
* **Advanced Machine Learning:** Exploring deeper neural networks or ensemble * learning techniques to potentially improve detection accuracy and robustness.
* **Severity Classification:** Extending the project beyond simple detection to identify the seerity of mildew infection, guiding treatment decisions.
* **Multi-Disease Detection:** Enlarging the model's capabilities to identify other common cherry leaf diseases, offering a broader diagnostic tool.
### Future Potential
* **Broader Application:** The project's core technology has the potential to be adapted to detect diseases in other crops facing similar challenges. This opens doors for broader application in the agricultural domain.
* **Continuous Improvement:** The project can be further enhanced through ongoing research and development, potentially leading to even higher accuracy and more advanced functionalities.

### Project Overview
This project tackles cherry leaf mildew detection using a combination of image analysis and machine learning. The goal is to develop a system that can accurately identify whether a cherry leaf is healthy or infected with powdery mildew, addressing the challenges faced by Farmy & Foods, a company specializing in cherry plantations. Their current manual inspection process is time-consuming and inefficient.

### Cherry Leaf Mildew Detection with Machine Learning
This project tackles cherry leaf mildew detection using a combination of image analysis and machine learning. The goal is to develop a system that can accurately identify whether a cherry leaf is healthy or infected with powdery mildew.

### Project Goals
* Develop data visualizations to differentiate healthy and mildewed cherry leaves visually.
* Build a machine learning model to predict mildew presence in cherry leaf images with an accuracy of **97%** (as specified by the client).

### Problem and Opportunity
**Problem:** Manually inspecting cherry orchards for powdery mildew is time-consuming and inefficient.
**Opportunity:** Develop an automated system using machine learning to analyze cherry leaf images for instant mildew detection.
### Business Requirements - Hypothesis
**Client Requirements:**
**Hypothesis 1: Visually Distinguishing Mildew:** A visual tool can effectively differentiate healthy and mildewed leaves based on visual cues like spots and discoloration.\
**Assumption:** We hypothesize that there are significant visual differences between healthy and mildewed cherry leaves that can be captured through image analysis. These differences may include variations in color (e.g., yellowing of leaves), presence of spots or patches, and overall leaf texture.\
**Validation Method 1:** User testing with cherry farmers to assess the tool's usability and accuracy in identifying mildew based on visual inspection.
* Analyze average healthy and mildewed leaf images to identify and highlight key visual discrepancies.
* Present variability images within each class to showcase the range of appearances for both healthy and mildewed leaves.

**Hypothesis 2: Machine Learning Model Effectiveness** The ML system can accurately identify mildew in cherry leaf images.\
**Assumption:** We hypothesize that a machine learning model trained on the cherry leaf image dataset can accurately predict the presence or absence of mildew in unseen cherry leaf images.\
**Validation Method 1:**
Performance metrics like accuracy, precision, recall, and F1-score calculated on a held-out test set.
* Split the dataset into training, validation, and test sets.
* Train the machine learning model on the training set and evaluate its performance on the validation set to fine-tune hyperparameters.
* Assess the model's final performance on the unseen test set using metrics like accuracy, precision, recall, and F1-score.

---
### Mapping Business Requirements to Data Tasks 
This section bridges the gap between the client's needs and the project's technical execution. Here, we'll explore how data visualizations and machine learning tasks directly address the client's specific requirements:\
**Client Requirement 1:** Visual Differentiation Study Goal: Empower cherry farmers with the ability to visually identify healthy and mildewed leaves. Data Tasks: Analyze the dataset to identify key visual characteristics that differentiate healthy and mildewed leaves (e.g., color variations, presence of spots). Create visualizations such as average healthy vs. mildewed leaf comparisons, variability images showcasing the range of appearances within each class, and image montages for broader reference.\
**Client Requirement 2:** Mildew Prediction Model.\
**Goal:** Develop a machine learning model that accurately predicts the presence or absence of mildew in unseen cherry leaf images.
### Data Tasks
* Preprocess the image data (e.g., resizing, normalization) for compatibility with the machine learning model.
* Train a machine learning model on the labeled dataset (healthy vs. mildewed).
* Evaluate the model's performance on a separate test set to ensure generalizability.
 
### Additional Considerations:

* **Scalability:** The system should handle analyzing images from a large number of trees across a farm.
* **Future Applications:** Potential for adapting the technology to detect diseases in other crops.

### Data
**Dataset Overview**\
The project leverages a dataset of cherry leaf images obtained from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves). This dataset contains of 4208 Number of cherry leaf images, that is categorized into two classes: healthy and mildewed leaves. This classification reflects the client's concerns about potential powdery mildew impacting product quality. By training the machine learning model on these images, we aim to equip cherry farmers with a tool for early detection and improved crop management.

**Dataset Description in Summary**
* 4208 cherry leaf images (50/50 split between healthy and infected)
* Machine Learning Model CNN
* Quantify the time saved by farmers through automated detection compared to manual inspection. 
* Consider potential cost savings from earlier disease identification and treatment. 
* Highlight the potential for increased crop yield and overall farm productivity.

---
### Business Requirements - Method and Models
**Visual Differentiation Study:** Develop a method to visually distinguish healthy cherry leaves from those infected with powdery mildew. This can be achieved through conventional data analysis techniques like creating average and variability images for both leaf classes and showcasing image montages.\

**Mildew Prediction Model:** Build a machine learning model to predict the presence (or absence) of powdery mildew in cherry leaf images. The target accuracy for this model is as mentioned before **97%**.

### Conventional Data Analysis Can Address Visual Differentiation
Conventional data analysis methods can effectively address the client's need for a visual differentiation study. By generating various image comparisons and visualizations, we can highlight the key characteristics that differentiate healthy and mildewed leaves.

### Client Needs a Dashboard for User Interaction
A dashboard is the most suitable solution for the client. It provides a user-friendly interface for interacting with the model, visualizing results, and exploring the visual differentiation findings.

### Project Success Criteria
**The client considers the project successful if it delivers two functionalities:**
* **Visual Differentiation Study:** A clear presentation of how to visually recognize healthy and mildewed cherry leaves.
* **Mildew Prediction Capability:** A system that accurately predicts the presence of mildew in cherry leaf images.

### Project Description
This project addresses the challenge of detecting powdery mildew in cherry leaves faced by Farmy & Foods, a company specializing in cherry plantations. The current manual inspection process is time-consuming and inefficient. 

### This project aims to develop
* **Data visualizations:** To differentiate healthy and mildewed cherry leaves visually.
* **Machine learning model:** To predict mildew presence in cherry leaf images with an accuracy of **97%** (as specified by the client).

---
### Business Requirements
**Business Requirement 1:** Visual Differentiation Study
This study aims to develop a clear method for visually distinguishing healthy cherry leaves from those infected with powdery mildew. This will be achieved through conventional data analysis techniques. Here's what the study will include:

**Average Images and Variability Images:** Creation of separate average images for both healthy and mildewed cherry leaves. These images represent the "typical" features of each class. Additionally, it will generate variability images to visualize the range of variations within each class. This will help users identify subtle differences that might not be apparent in the average images alone.

**Comparison of Average Images:** An analyzis and a presentation of the key visual differences between the average healthy and average mildewed cherry leaf images. This comparison will highlight specific characteristics that can be used for rapid visual identification.

**Image Montages:** Separate image montages will be created for both healthy and mildewed leaf classes. These montages will showcase a collection of representative images from each class, providing users with a broader visual reference.

**Business Requirement 2:** Mildew Prediction Model
This requirement focuses on building a machine learning system that can automatically predict the presence of powdery mildew in cherry leaf images.

---
### The considerations for the Business Requirement 2 component
**Neural Network Approach:** For this project it is recommended utilizing a Neural Network, specifically a Convolutional Neural Network (CNN), for this task. CNNs are well-suited for image classification problems like this, as they can effectively learn the relationships between image features (e.g., color, texture) and the corresponding labels (healthy or mildewed).

**Image Size and Model Size Trade-off:** While using the original image size of 256x256 pixels would likely lead to a more accurate model, it would also result in a larger model size exceeding 100Mb. To facilitate a smoother push to GitHub, consider using a smaller image size like 100x100 or 50x50. This might slightly impact accuracy, but you can experiment to find the optimal balance between performance and model size.

**Git LFS for Large Models (if applicable):** If the final model size exceeds 100Mb even with a reduced image size, you can use Git Large File Storage (Git LFS) to efficiently manage and push the model to GitHub.

**Remember:** The specific techniques and visualizations used may be adjusted based on the data exploration findings.

## Cherry Leaf Mildew Detection Dashboard: Empowering Cherry Farmers
The project dashboard serves and delivers insights and functionality as the primary user interface for interacting with the system and visualizing results. The project dashboard serves as the central hub for cherry farmers to interact with the system, gain insights, and leverage automated mildew detection.\
### Here's a breakdown of its functionalities
**Model Performance Metrics:**

This section caters to users interested in the technical aspects of the model (e.g., researchers). It provides details on the model's performance on the unseen test set, potentially including metrics like accuracy, precision, recall, and F1-score. By incorporating these functionalities, the dashboard empowers cherry farmers to not only understand mildew visually but also leverage the automated detection system for informed decision-making within their orchards. 

**Cherry Leaf Mildew Detection with CNN:**

This project utilizes a Convolutional Neural Network (CNN) to automate cherry leaf mildew detection for Farmy & Foods, a cherry plantation company. Their current manual inspection method is inefficient. The CNN analyzes cherry leaf images to differentiate between healthy and mildewed leaves. Researchers evaluate the model's performance on a separate test set using metrics like accuracy, precision, recall, and F1-score.

**How the CNN Detects Mildew on Cherry Leaves:**

Image Preprocessing: The cherry leaf images are likely preprocessed before feeding them into the CNN. This might involve resizing, normalization, and potentially color space conversion for better model performance.

Feature Extraction: The CNN's convolutional layers then extract features from the images. These features capture patterns and details relevant to differentiating healthy and mildewed leaves. 

**For example, the CNN might learn to identify features like:**

* **Color variations:** Mildewed leaves often exhibit discoloration or white powdery spots.
* **Texture changes:** Mildew can cause a rougher texture on the leaf surface.
* **Leaf shape alterations:** Mildew can distort the leaf's normal shape.
* **Classification:** As the CNN processes the image through its layers, it learns to classify the leaf based on the extracted features. The final layers use functions like softmax to assign a probability score to each class (healthy or mildewed). The leaf is classified based on the class with the highest probability.

**Metrics Used for Evaluation:**

* **Accuracy:** This is the most common metric, representing the proportion of images the model correctly classifies (healthy or mildewed).
**Precision:** This measures how many of the images predicted as mildewed are truly mildewed. It avoids classifying healthy leaves as mildewed (reducing false positives).
* **Recall:** This measures how many of the actual mildewed leaves are correctly identified by the model. It avoids missing mildewed leaves (reducing false negatives).

* **F1-Score:** This metric combines precision and recall into a single score, providing a balance between the two. A high F1-score indicates the model performs well in both identifying true positives (mildewed leaves) and avoiding false positives (healthy leaves classified as mildewed).

These metrics provide a comprehensive picture of the model's effectiveness in detecting cherry leaf mildew. By achieving a high accuracy, precision, recall, and F1-score, the CNN can be a reliable tool for automated disease detection in cherry orchards.

**Conclusion:**
This project successfully leveraged a Convolutional Neural Network (CNN) to automate cherry leaf mildew detection for Farmy & Foods. The CNN effectively differentiates between healthy and mildewed leaves by extracting key visual features from cherry leaf images. Evaluation metrics like accuracy, precision, recall, and F1-score ensure the model's reliability in real-world applications.

The user-friendly interface provides a practical tool for farmers, significantly reducing inspection time and enabling earlier disease identification. This translates to increased farm productivity through minimized crop damage and improved treatment strategies.

Looking ahead, the project's potential extends beyond the current implementation. Future enhancements like mobile app integration, advanced machine learning techniques, and broader disease detection capabilities hold promise for further streamlining disease management in cherry orchards. Additionally, deploying the model as an API could unlock its benefits for a wider range of stakeholders in the agricultural sector.

Overall, this project demonstrates the power of CNNs in automating disease detection tasks. By combining image analysis with machine learning, it offers a valuable solution for improving efficiency and productivity in cherry farming.

### Data Description
The data used in this project consists of cherry leaf images obtained from [Data source link (if publicly available)]. The images are in PNG format and have a resolution of 256x256 pixels. Using a smaller image size (e.g., 100x100) might be beneficial for reducing model size and facilitating deployment on platforms with storage limitations. Large models can be pushed to GitHub using Git LFS.

### Model Details
This project utilizes a Convolutional Neural Network (CNN) model for binary classification. The model is trained to distinguish between healthy and mildewed cherry leaves based on the image features it extracts.

### Dashboard Description
The project includes a Streamlit dashboard that provides users with various functionalities to address the client's requirements.

### Visual Differentiation
This section demonstrates how the project's data analysis and model development directly translate to fulfilling the client's requirements for visual differentiation and automated mildew prediction.

* **Visual Differentiation Study:** Data visualizations like average images, variability images, and image montages will help users distinguish healthy and mildewed leaves.
* **Mildew Prediction Model:** The machine learning model will be trained to analyze image features and predict the presence of mildew.

## ML Business Case for Farmy & Foods
The developed machine learning model presents a compelling business case for Farmy & Foods, offering significant advantages over their current manual inspection methods for cherry leaf mildew detection:

* **Enhanced Efficiency:** Automating mildew detection with the model significantly reduces the time and resources required compared to manual inspection. Farmy & Foods can potentially save 30 minutes per tree, translating to faster and more comprehensive orchard scouting.

* **Improved Product Quality Control:** Early and accurate detection of mildew enables Farmy & Foods to take timely action, such as targeted fungicide application. This minimizes the spread of disease and safeguards the overall quality of their cherry crops.

* **Reduced Operational Costs:** Replacing manual labor with an automated system leads to cost savings in the long run. Farmy & Foods can potentially reallocate resources currently dedicated to manual inspection towards other areas of farm management.

* **Scalability for Future Growth:** The ML model can readily scale to accommodate the vast number of trees and multiple farms managed by Farmy & Foods. This ensures efficient mildew detection even as their operations expand. 

By leveraging this machine learning solution, Farmy & Foods can gain a significant competitive edge in terms of operational efficiency, product quality control, and cost-effectiveness. This translates to long-term benefits for their business and the overall sustainability of their cherry farming practices.

## Dashboard Design: Empowering Cherry Farmers
The user interface (UI) of the project dashboard is designed to be user-friendly and informative for cherry farmers. It provides a central hub for interacting with the system, gaining insights, and leveraging automated mildew detection. By incorporating these functionalities, the dashboard empowers cherry farmers to not only understand mildew visually but also leverage the automated detection system for informed decision-making within their orchards. 

## The Streamlit dashboard addresses the client's needs:

### Page 1: Quick Project Summary
* Introduceas the project and target audience (e.g., cherry farmers).
* Summarizes cherry leaf mildew and its challenges.
* Describes the dataset (size, source, image types).
* Outlines business requirements (visual differentiation and image-based prediction).
* Includes links for further learning about cherry leaf mildew.

### Page 2: Project Hypothesis and Validation
* Explains the hypothesis for each business requirement.
* Details the validation methods used.
* Summarizes the validation results.

### Page 3: Mildew Visualizer (for Business Requirement 1)
* Provides checkboxes for visualizations.
* Differences between average and standard deviation images (infected vs. uninfected).
* Comparisons of average infected and uninfected leaf images.
* Image montages for infected or uninfected leaves.

### Page 4: Mildew Detector (for Business Requirement 2)
* Uploader for multiple cherry leaf images.

**Displays uploaded images with:**
* Prediction statements (presence/absence of mildew) and probabilities.
* Summarizes results in a downloadable table.

### Page 5: ML Performance Metrics
* Shows label frequencies (infected/uninfected) across datasets (training, validation, test).
* Visualizes the model's learning process (accuracy and loss over training epochs).
* Summarizes model performance on the test set using metrics like accuracy, precision, recall, and F1-score.

**Overall,** this design ensures a user-friendly and informative dashboard that meets the client's requirements and facilitates data exploration, model evaluation, and actionable insights generation.

## Deployment: APP deployed on Heroku
[Mildew Detection in Cherry Leaves](https://mildewdetectionincherryleaves-c24b4ea4a636.herokuapp.com/)

The Streamlit app i deployed to Heroku, and the model can be tested in this deployment.

## How to Use This Repository
* Fork this repository on GitHub.
* Clone your forked repository to your local machine.
* Set up your development environment:
* Install required libraries (refer to the requirements.txt file).
* Consider using a virtual environment for managing dependencies.

### Run the project
* Navigate to the project directory in your terminal.
* Run streamlit run app.py to launch the Streamlit dashboard.
* Access Jupyter Notebooks:
* Open the "jupyter_notebooks" directory.
* Select and run the desired notebook.

### Testing
**Comprehensive Testing Ensures Accuracy:** To guarantee the reliability and functionality of the cherry leaf mildew detection project, comprehensive testing was undertaken.  Not only were all features rigorously evaluated within the Streamlit app environment, mirroring the intended user experience, but each step of the code was also manually tested in a Jupyter Notebook to validate its logic and performance. 

**Additionally, a crucial aspect of the testing process involved manually testing a diverse set of cherry leaf images**, ensuring the machine learning functions accurately classified both healthy and mildew-infected leaves.

### Unfixed Bugs
During development and testing, no critical bugs were identified that would impede the core functionalities of the system. Hence, there was issues within the code that needed to be adjusted fore the packages used.

### Deployment Considerations: 
Heroku was initially considered for deployment, but limitations arose due to model size and package dependencies. For similar projects with larger requirements, consider platforms designed specifically for machine learning deployments. Initial deployment attempts were made on Heroku, but limitations were encountered due to model size and package dependencies. 

Therefore, consider deploying on platforms designed for machine learning projects with larger requirements. While initial deployment attempts were made on Heroku, challenges arose due to the platform's soft limit of 300MB and hard limit of 500MB. The combined size of necessary packages, including Streamlit and TensorFlow-CPU, alongside project files, exceeded this limit. 

While Heroku support allowed a temporary increase to 600MB, it's important to note that this approach is not recommended for long-term use due to potential performance impacts. As a result, for similar machine learning projects with larger dependencies, deploying on alternative platforms designed for these needs is recommended.

---
### Main Data Analysis and Machine Learning Libraries
**This project leverages several Python packages to achieve its functionality:**

**tensorflow-cpu:** This core deep learning library from Google enables the creation, training, and deployment of machine learning models, specifically a Convolutional Neural Network (CNN) for this project. The tensorflow-cpu variant is used for CPU-based execution, while tensorflow-gpu leverages a GPU (if available) for faster computations.

**streamlit:** This library simplifies the creation of interactive web applications. It's used to develop the user-friendly Streamlit dashboard that serves as the primary interface for interacting with the model and visualizing results.\

**h5py:** This package facilitates working with Hierarchical Data Format 5 (HDF5) files, a common format for storing and managing large datasets, including images and model weights. It is used for loading and saving the trained CNN model if needed.\

**numpy:** This fundamental library provides powerful array manipulation capabilities. It's essential for numerical computations, image preprocessing (e.g., resizing, normalization), and working with matrices in the CNN model.\

**pandas:** This data analysis library offers versatile tools for working with tabular data (DataFrames). It is used for tasks like exploring and cleaning the image metadata (e.g., labels) associated with the cherry leaf dataset.\

**matplotlib:** This visualization library is a cornerstone for creating various plots and charts. It might be used to generate the visualizations for the mildew visualizer component of the dashboard, such as comparisons between healthy and mildewed leaf images.\

**pillow (PIL Fork):** This Python Imaging Library (PIL) fork provides functionalities for image processing tasks like loading, resizing, manipulating, and saving image files in various formats (e.g., PNG, JPEG). It's used for image preprocessing before feeding them into the CNN model.

**seaborn:** This library builds on top of matplotlib, offering a high-level interface for creating statistical data visualizations with a focus on aesthetics. It could be used to create more visually appealing charts and plots for the dashboard.

**keras:** This deep learning API, often used in conjunction with TensorFlow, simplifies the creation and training of neural network models. While TensorFlow provides the core building blocks, Keras offers a more user-friendly interface for building and configuring the CNN model in this project.

**keras:** This deep learning API, often used in conjunction with TensorFlow, simplifies the creation and training of neural network models. While TensorFlow provides the core building blocks, Keras offers a more user-friendly interface for building and configuring the CNN model in this project.

**joblib:** This package is used for saving and loading Python objects, allowing you to persist the trained CNN model for future use without retraining. It can be helpful for saving the model after training and loading it in the Streamlit app to make predictions on new cherry leaf images.

**plotly** (optional): This library can be used to create interactive visualizations, particularly useful for exploring data with many dimensions. It could be an alternative or complement to matplotlib and seaborn, especially if you want users to interact with the visualizations dynamically in the dashboard.

---

### Credits
This project is an original creation based on turorial and bricklaying code fron Code Institute, and the code used has been to some extend been remodeled, restructured, or written from scratch, specifically for this project. Foundational concepts and learning materials were sourced from Code Institute's Machine Learning course, and the project structure aligns with the course's learning objectives/goals.

---
### Content
**Wikipedia offered 2 links within this project:**

[Cherry leaf spot](https://en.wikipedia.org/wiki/Cherry_leaf_spot)\
[List of Dataset for Machine Learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)

---
### Media
There is no additonal media to this project to credit other than the [cherry-leaves](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) dataset provided by Code Institute.

---
### Conclusion: A Sustainable Solution for the Future of Farming
This project offers a compelling business case for cherry farmers. By leveraging machine learning for automated cherry leaf mildew detection, it presents a cost-effective, scalable, and accurate solution. This translates to significant improvements in farm efficiency, crop yield, and overall profitability. Furthermore, the potential for broader application in agriculture highlights the project's positive impact on sustainable farming practices for the future.

---
