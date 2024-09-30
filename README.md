# Streamlit-Work-App-2 :

🫀 Heart Disease Prediction App 💖 :

🚀 https://app-work-app-2-6b84scxqzpiyvhjlcrui9a.streamlit.app/

💻 

Welcome to the Heart Disease Prediction App! This Streamlit web application predicts the likelihood of heart disease based on user-inputted medical parameters.

🌟 Project Overview :

Heart disease is a leading cause of mortality worldwide, accounting for millions of deaths annually. This application aims to provide users with an easy way to assess their heart health

risk through a predictive model. By inputting personal and medical information, users can receive predictions on their likelihood of having heart disease.

📊 Dataset
The model is trained on a dataset containing various medical parameters related to heart health. The data includes the following features:

👤 Age: Age of the patient

🧍 Gender: Gender of the patient (Male/Female)

💓 Chest Pain Type: Type of chest pain experienced

🩻 Resting ECG: Results of the resting electrocardiogram

🍔 Cholesterol (mg/dL): Cholesterol level in mg/dL

🩸 Fasting Blood Sugar: Fasting blood sugar levels

🫀 Resting Blood Pressure (mmHg): Blood pressure at rest

🏃 Max Heart Rate: Maximum heart rate achieved

🏋️‍♂️ Exercise Induced Angina: Whether the patient experiences angina during exercise

📉 Old Peak (ST Depression): ST depression induced by exercise relative to rest

📈 ST Slope: Slope of the peak exercise ST segment

The dataset used for training is included in this repository as heart.csv.

🛠 Features :

Input Form :

User-Friendly Interface: Enter your personal and medical details using an intuitive form. ✍️

Predictions :

Risk Assessment: The app predicts whether you are at risk of heart disease. 🔍

Visual Feedback :

Image Representation: Displays images of a healthy heart ❤️ and a sick heart 💔 based on the prediction.

Prediction Probabilities :

Probability of Being at Risk: Indicates the likelihood of having heart disease. 📈

Probability of Not Being at Risk: Indicates the likelihood of not having heart disease. 📉

📚 Libraries Used :

The application utilizes the following Python libraries:

Pandas: For data manipulation and analysis. 📊

Scikit-learn: For building and training the machine learning model. 🤖

Streamlit: For creating the web application interface. 🌐

⚙️ Model :

The app employs a Random Forest Classifier to predict the likelihood of heart disease. Random Forest is chosen for its robustness and effectiveness in handling classification tasks,

especially with datasets containing categorical variables.

Model Training Steps 

Data Preprocessing:

Label encoding is used to transform categorical variables into numerical formats suitable for model training. 🔄

Data Splitting:

The dataset is divided into training and testing sets to evaluate the model's performance. 🔍

Model Training:

The Random Forest model is trained using the training dataset. ⚙️

Prediction:

User inputs are processed to ensure consistency with the model's feature set before making predictions. 📝

🚀 How to Use the App :

Enter Your Information: Input your medical details in the sidebar. 📝

Get Prediction: Click the "🔮 Predict My Heart Health!" button. 🚀

View Results: The app will display whether you are at risk of heart disease, along with relevant probabilities and images. 🖼️

🎯 Results :

Upon clicking the predict button, users will receive:

A prediction stating whether they are at risk of heart disease. ⚠️

The corresponding probabilities of being at risk and not being at risk. 📊

Images representing the health status of the heart based on the prediction. ❤️💔

🌈 Future Work :

Future improvements may include:

Implementing additional advanced models to enhance prediction accuracy. 🔍

Adding user authentication for personalized tracking of health data. 🔑

Providing educational resources and articles about heart health and disease prevention. 📚

🤝 Contact :

For questions or feedback, please reach out to:

Name: Oumayma Oueslati 🌟
LinkedIn: https://www.linkedin.com/in/oumayma-oueslati-12a5462b2/ 🔗
Email: oueslatioumayma157@gmail.com 📧
