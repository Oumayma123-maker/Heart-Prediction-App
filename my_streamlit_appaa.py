
# import libraries
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing

# Load and preprocess the data
data = pd.read_csv('heart.csv')  # Import data
data1 = data.copy()  # Copy of data

# Data Transformation (label encoding)
label_encoder = preprocessing.LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
mapping_dict_sex = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['ChestPainType'] = label_encoder.fit_transform(data['ChestPainType'])
mapping_dict_chest = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['RestingECG'] = label_encoder.fit_transform(data['RestingECG'])
mapping_dict_rest = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['ExerciseAngina'] = label_encoder.fit_transform(data['ExerciseAngina'])
mapping_dict_ex = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['ST_Slope'] = label_encoder.fit_transform(data['ST_Slope'])
mapping_dict_slope = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Split the dataset
x = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

# Set the background color using CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ff7e5f, #feb47b); /* Gradient background */
        color: white; /* Change text color to white */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title and sidebar inputs
st.title("🫀 How is your heart? 💖")
st.write("Enter Your Information Below to Check Your Heart's Health: 🩺")

st.markdown("---")  # Add a horizontal line for separation

# Input fields in the sidebar
st.sidebar.header("Your Information")
age = st.sidebar.number_input('👤 Age:')
sex = st.sidebar.selectbox('🧍 Gender:', data1['Sex'].unique())
sex_map = mapping_dict_sex[sex]

chest = st.sidebar.selectbox('💓 Chest Pain Type:', data1['ChestPainType'].unique())
chest_map = mapping_dict_chest[chest]

resting = st.sidebar.selectbox('🩻 ECG:', data1['RestingECG'].unique())
resting_map = mapping_dict_rest[resting]

cholesterol = st.sidebar.number_input('🍔 Cholesterol (mg/dL):')
fasting = st.sidebar.number_input('🩸 Fasting Blood Sugar (mg/dL):')
resting_blood = st.sidebar.number_input('🫀 Resting Blood Pressure (mmHg):')
max_hr = st.sidebar.number_input('🏃 Max Heart Rate:')
ex = st.sidebar.selectbox('🏋️‍♂️ Exercise Angina:', data1['ExerciseAngina'].unique())
ex_map = mapping_dict_ex[ex]

old_peak = st.sidebar.number_input('📉 Old peak (ST Depression):')
slope = st.sidebar.selectbox('📈 ST Slope:', data1['ST_Slope'].unique())
slope_map = mapping_dict_slope[slope]

# Prepare data for prediction
my_dict = {
    'Age': age, 'Sex': sex_map, 'ChestPainType': chest_map,
    'RestingECG': resting_map, 'Cholesterol': cholesterol,
    'FastingBS': fasting, 'RestingBP': resting_blood,
    'MaxHR': max_hr, 'ExerciseAngina': ex_map, 'Oldpeak': old_peak, 'ST_Slope': slope_map
}

data_test = pd.DataFrame(my_dict, index=[0])

# Ensure feature order consistency between training and prediction
data_test = data_test[x_train.columns]

# Predict button and model prediction
if st.button('🔮 Predict My Heart Health!'):
    prediction = clf.predict(data_test)
    prediction_proba = clf.predict_proba(data_test)

    # Display prediction result
    st.subheader('✨ Prediction Result:')
    if prediction[0] == 1:
        st.write("⚠️ **You are at risk of heart disease.** 💔 Please consult a doctor for further guidance.")
        # Display sick heart image
        st.image("sick heart.jpg", caption="Sick Heart", use_column_width=True)
    else:
        st.write("🎉 **You are not at risk of heart disease.** 💖 Keep up with your healthy lifestyle!")
        # Display healthy heart image
        st.image("healthy heart.png", caption="Healthy Heart", use_column_width=True)

    # Display prediction probabilities
    st.subheader('📊 Prediction Probability:')
    st.write(f"💔 Probability of being at risk: **{prediction_proba[0][1]*100:.2f}%**")
    st.write(f"💖 Probability of not being at risk: **{prediction_proba[0][0]*100:.2f}%**")

    # Add some nice footer or extra message
    st.markdown("---")
    st.write("💡 *Note: This prediction is for educational purposes only and should not replace professional medical advice.*")
