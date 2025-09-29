# streamlit_student_prediction.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Student Marks/CGPA Prediction System")

# 1. CSV upload or manual input
uploaded_file = st.file_uploader("Upload Student CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.write("Enter student details manually:")
    name = st.text_input("Name")
    roll = st.text_input("Roll Number")
    course = st.selectbox("Course", ["B.Tech", "BBA", "BCA", "Medical", "Other"])
    cgpa = st.number_input("Semester Avg CGPA", min_value=0.0, max_value=10.0, step=0.1)
    study_hours = st.number_input("Study Hours per week", min_value=0)
    attendance = st.number_input("Attendance %", min_value=0, max_value=100)
    assignments = st.number_input("Assignments Completed", min_value=0)
    extra = st.number_input("Extra Activities", min_value=0)

    df = pd.DataFrame({
        "Name": [name],
        "RollNumber": [roll],
        "Course": [course],
        "SemesterAvgCGPA": [cgpa],
        "StudyHours": [study_hours],
        "AttendancePercentage": [attendance],
        "AssignmentsCompleted": [assignments],
        "ExtraActivities": [extra]
    })

# 2. Encode Course for model
df_encoded = df.copy()
df_encoded["Course"] = df_encoded["Course"].map({
    "B.Tech":1, "BBA":2, "BCA":3, "Medical":4, "Other":5
})

# 3. Dummy model for prediction (Linear Regression)
# y = SemesterAvgCGPA + 0.05*StudyHours + 0.03*Attendance + 0.02*Assignments + 0.01*ExtraActivities
X = df_encoded[["SemesterAvgCGPA","StudyHours","AttendancePercentage","AssignmentsCompleted","ExtraActivities"]]
y = df_encoded["SemesterAvgCGPA"] + 0.05*df_encoded["StudyHours"] + 0.03*df_encoded["AttendancePercentage"] + 0.02*df_encoded["AssignmentsCompleted"] + 0.01*df_encoded["ExtraActivities"]

model = LinearRegression()
model.fit(X, y)
df["PredictedMarks"] = model.predict(X).round(2)

# 4. Show top 10 students
st.subheader("Top 10 Students")
top_students = df.sort_values(by="PredictedMarks", ascending=False).head(10)
st.dataframe(top_students[["Name","RollNumber","Course","PredictedMarks"]])

# 5. Option to download CSV
st.download_button(
    label="Download CSV with Predictions",
    data=top_students.to_csv(index=False),
    file_name='top_students.csv',
    mime='text/csv'
)
