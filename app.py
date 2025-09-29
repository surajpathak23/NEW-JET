import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import io
import plotly.express as px

# --- Page Configuration and CSS ---
st.set_page_config(
    page_title="Student CGPA Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for a better look
st.markdown("""
<style>
.main-header {
    font-size: 3em;
    font-weight: bold;
    color: #008CBA;
    text-align: center;
    padding-bottom: 20px;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.stTable {
    font-size: 14px;
}
.st-emotion-cache-1cypcdp {
    color: #4CAF50;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='main-header'>🎓 Student CGPA Predictor</div>", unsafe_allow_html=True)

# --- Sidebar for File Upload & About ---
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.info("Upload your student dataset here. The file should have 'Study Hours', 'Previous CGPA', and 'Final CGPA' columns.")
    
    st.markdown("---")
    st.header("About This Project")
    st.write("This project uses a **Linear Regression** model to predict a student's final CGPA based on their study hours and previous semester's CGPA.")
    st.write("The model learns from the uploaded dataset to provide accurate predictions.")

# --- Main App Logic ---
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        required_cols = ['Study Hours', 'Previous CGPA', 'Final CGPA']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: The uploaded CSV must contain the following columns: {', '.join(required_cols)}")
            df = None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None

if df is not None:
    # --- Dataset Summary ---
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Students", value=len(df))
    with col2:
        if 'Final CGPA' in df.columns:
            avg_cgpa = df['Final CGPA'].mean()
            st.metric(label="Average Final CGPA", value=f"{avg_cgpa:.2f}")

    st.markdown("---")

    # --- Top 10 Students Section ---
    st.header("Top 10 Students (from Uploaded Data)")

    if 'Final CGPA' in df.columns:
        top_students = df.sort_values(by='Final CGPA', ascending=False).head(10)
        st.table(top_students.reset_index(drop=True))
    else:
        st.warning("Cannot display top students. 'Final CGPA' column is missing in the uploaded file.")

    st.markdown("---")
    
    # --- Data Preprocessing and Model Training ---
    try:
        features = ['Study Hours', 'Previous CGPA']
        target = 'Final CGPA'
        
        X = df[features]
        y = df[target]

        # Splitting data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
    except KeyError:
        st.error("Missing required columns for model training. Please check your dataset.")
        model = None
        
    # --- Prediction Section ---
    st.header("Predict a New Student's CGPA")
    if model:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Student Name", placeholder="Enter student's full name")
                roll_number = st.text_input("Roll Number", placeholder="Enter roll number")
            
            with col2:
                study_hours = st.slider(
                    "Study Hours (per day)", 
                    min_value=0.0, 
                    max_value=12.0, 
                    value=5.0, 
                    step=0.1, 
                    help="Average hours a student studies per day."
                )
                previous_cgpa = st.slider(
                    "Previous Semester CGPA",
                    min_value=0.0,
                    max_value=10.0,
                    value=7.5,
                    step=0.1,
                    help="CGPA from the previous semester."
                )

            submit_button = st.form_submit_button("Predict CGPA")

        if submit_button:
            new_student_data = pd.DataFrame({
                'Study Hours': [study_hours],
                'Previous CGPA': [previous_cgpa]
            })
            prediction = model.predict(new_student_data)[0]
            final_prediction = np.clip(prediction, 5.0, 10.0)
            st.success(f"### Predicted CGPA for {name}: **{final_prediction:.2f}**")

    st.markdown("---")

    # --- Model Performance & Visualization Section ---
    st.header("Model Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Performance")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", help="Average difference between predicted and actual CGPA.")
        st.metric("R-squared Score (R²)", f"{r2:.2f}", help="Indicates how well the model's predictions fit the data. A value closer to 1 is better.")
    
    with col2:
        st.subheader("CGPA vs. Study Hours")
        fig = px.scatter(df, x='Study Hours', y='Final CGPA',
                         title='Relationship between Study Hours and CGPA',
                         labels={'Study Hours': 'Study Hours (per day)', 'Final CGPA': 'Final CGPA'},
                         hover_data=['Name', 'Previous CGPA'])
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file from the sidebar to start the prediction.")
