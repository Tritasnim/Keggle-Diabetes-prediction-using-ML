# Install necessary libraries
# pip install streamlit pandas matplotlib plotly scikit-learn seaborn

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load Dataset
df = pd.read_csv(r'D:\streamlit\diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to Collect User Input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 300, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 130, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 900, 79)
    bmi = st.sidebar.slider('BMI', 0, 70, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.47)
    age = st.sidebar.slider('Age', 21, 100, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

from sklearn.linear_model import LogisticRegression

# MODEL
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
user_result = log_reg.predict(user_data)

from sklearn.svm import SVC

# MODEL
svc = SVC(probability=True)
svc.fit(x_train, y_train)
user_result = svc.predict(user_data)

from xgboost import XGBClassifier

# MODEL
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_train, y_train)
user_result = xgb.predict(user_data)

# VISUALIZATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
color = 'red' if user_result[0] == 1 else 'blue'

# Age vs Pregnancies
st.header('Pregnancy Count Graph (Others vs Yours)')
fig_preg = plt.figure()
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_preg)

# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_glucose)

# Age vs Blood Pressure
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color)
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_bp)

# OUTPUT
st.subheader('Your Report:')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# Accuracy
st.subheader('Accuracy:')
st.write(f"Accuracy for randomforest:{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")
st.write(f"Accuracy for logistic regression:{accuracy_score(y_test, log_reg.predict(x_test)) * 100:.2f}%")
st.write(f"Accuracy for svm:{accuracy_score(y_test, svc.predict(x_test)) * 100:.2f}%")
st.write(f"Accuracy for svm:{accuracy_score(y_test, xgb.predict(x_test)) * 100:.2f}%")

