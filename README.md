# 🎓 Student CGPA Predictor

> Streamlit-based web application to predict a student's final CGPA using a **Linear Regression** model, based on their **study hours** and **previous semester's CGPA**.

## ✨ Features

  * **Dataset Upload:** Users can upload their own student data (`.csv` format). The required columns are **'Study Hours'**, **'Previous CGPA'**, and **'Final CGPA'**.
  * **Dataset Overview:** Displays key statistics like **Total Students** and **Average Final CGPA**.
  * **Top Students:** Shows a table of the top performing students from the uploaded data.
  * **CGPA Prediction:** Allows users to input a new student's details (**Study Hours** and **Previous CGPA**) to get a predicted **Final CGPA**.
  * **Model Insights:**
      * **Model Performance** metrics (e.g., Mean Absolute Error (MAE) and R-Squared ($\text{R}^2$)).
      * **Visualization** of the relationship between **Study Hours** and **Final CGPA**.

## 🛠️ Technology Stack

  * **Python**
  * **Streamlit:** For creating the interactive web application.
  * **Pandas:** For data handling and manipulation.
  * **Scikit-learn (sklearn):** For implementing the **Linear Regression** model.
  * **Matplotlib / Plotly:** For data visualization.

## 🚀 How to Run Locally

Follow these steps to set up and run the application on your local machine.

### Prerequisites

You need **Python** installed on your system.

### 1\. Clone the repository

```bash
git clone https://github.com/surajpathak23/NEW-JET # Example path based on your screenshots
cd NEW-JET # Replace with your project folder name
```

### 2\. Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
# (Assuming you have a requirements.txt file with streamlit, pandas, sklearn, etc.)
```

### 3\. Run the Streamlit Application

```bash
streamlit run <your_main_app_file>.py
# (e.g., streamlit run app.py)
```

The application will open automatically in your web browser.

## 📊 Model Performance

The Linear Regression model is trained on the uploaded dataset. Below are the key performance metrics from the example run:

| Metric | Value |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.26 |
| **R-Squared ($\text{R}^2$)** | 0.85 |

*A higher $\text{R}^2$ value (close to 1.0) and a lower MAE indicate a good fit for the model.*

## 📸 Application Screenshots

| Dataset Overview & Top Students | Prediction Interface | Model Insights |
| :---: | :---: | :---: |
| ![Dataset Overview & Top Students](https://github.com/surajpathak23/NEW-JET/blob/60734de117b62703cdb4f813fad4fd7a040530c1/Image/Dataset%20Overview%20%26%20Top%20Students.png?raw=true) | ![Prediction Interface](https://github.com/surajpathak23/NEW-JET/blob/60734de117b62703cdb4f813fad4fd7a040530c1/Image/Prediction%20Interface.png?raw=true) | ![Model Insights](https://github.com/surajpathak23/NEW-JET/blob/60734de117b62703cdb4f813fad4fd7a040530c1/Image/Model%20Insights.png?raw=true) |



-----

## 👨‍💻 Author

| Platform | Profile Link |
| :--- | :--- |
| **GitHub** | [surajpathak23](https://github.com/surajpathak23) |
| **LinkedIn** | [Suraj Kumar](https://www.linkedin.com/in/suraj-kumar-2307skp/) |
