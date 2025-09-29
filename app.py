import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Mark Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Load and Process Data ---
@st.cache_data
def load_data():
    try:
        # Load the CSV file from the same directory as the script
        df = pd.read_csv('student_data.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'student_data.csv' not found. Please make sure the file is in the same folder as app.py.")
        return None

df = load_data()

if df is not None:
    # Preprocessing the data
    features = ['Course', 'Semester Avg CGPA', 'Study Hours']
    target = 'Final Marks'

    X = df[features]
    y = df[target]

    # Use OneHotEncoder for the categorical 'Course' column
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), ['Course'])
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)

    # Train the model
    model = LinearRegression()
    model.fit(X_processed, y)

    # --- Streamlit Application Layout ---
    st.title("🎓 Student Mark Prediction Project")
    st.markdown("Enter student details to predict their marks and see the top performers.")
    
    st.markdown("---")

    # --- Prediction Section ---
    st.header("Predict a Student's Marks")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Student Name")
            roll_number = st.text_input("Roll Number")
        
        with col2:
            course = st.selectbox(
                "Select Course",
                options=['B.Tech', 'BBA', 'BCA', 'Medical', 'Other']
            )
            semester_avg_cgpa = st.slider("Semester Avg CGPA", 0.0, 10.0, 7.5, 0.1)
            study_hours = st.slider("Study Hours (per day)", 0, 10, 4)

        submit_button = st.form_submit_button("Predict Marks")

    if submit_button:
        # Create a new data point for prediction
        new_student_data = pd.DataFrame({
            'Course': [course],
            'Semester Avg CGPA': [semester_avg_cgpa],
            'Study Hours': [study_hours]
        })

        # Preprocess the new data using the same preprocessor
        new_student_processed = preprocessor.transform(new_student_data)

        # Predict the marks
        prediction = model.predict(new_student_processed)[0]

        st.success(f"### Predicted Marks for {name}: **{prediction:.2f}**")

    st.markdown("---")

    # --- Top 10 Students Section ---
    st.header("Top 10 Students (Based on Final Marks)")

    # Sort the dataframe by 'Final Marks' in descending order and get the top 10
    top_students = df.sort_values(by='Final Marks', ascending=False).head(10)

    # Display the top students in a table
    st.table(top_students[['Name', 'Roll Number', 'Course', 'Final Marks']].reset_index(drop=True))
