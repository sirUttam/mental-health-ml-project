# 🧠 Mental Health Treatment Prediction (ML Project)

This is a Machine Learning classification project that predicts whether a person may require mental health treatment based on lifestyle, work environment, behavioral patterns, and psychological history.

##-----------------------------------------------------------------------------------

## 🚀 Live Demo

👉 Access the deployed app here:  
https://check-your-mental-health.streamlit.app/


##-----------------------------------------------------------------------------------

## 🎯 Objective

To predict **treatment (Yes/No)** using real-world mental health related features such as stress, habits, work interest, and social behavior.

##-----------------------------------------------------------------------------------

## 📊 Dataset

- Rows: 292,364  
- Columns: 15 features  
- Target: `treatment`

##-----------------------------------------------------------------------------------

### Input Features (X)

- gender  
- country  
- occupation  
- self_employed  
- family_history  
- days_indoors  
- growing_stress  
- changes_habits  
- mental_health_history  
- mood_swings  
- coping_struggles  
- work_interest  
- social_weakness  
- mental_health_interview  
- care_options  

##-----------------------------------------------------------------------------------

### Target (y)

- `treatment`
  - 0 → Treatment not needed (NO)  
  - 1 → Treatment needed (YES)

##-----------------------------------------------------------------------------------

## 🤖 Machine Learning Approach

- Problem Type: Binary Classification  
- Model: *Mental_Health_Project.pkl*  
- Preprocessing:
  - Transformed columns by deletion and datatype conversion
  - Label encoding for categorical variables using LabelEncoder, cat.codes 
  - Handling encoded numerical features  
- Model saved using `joblib`

##-----------------------------------------------------------------------------------

## 🧰 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  

##-----------------------------------------------------------------------------------

## 📁 Project Structure
mental-health-project/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── models/
│ └── Mental_Health_Project.pkl
│ └── Label_Encoder.pkl
|
├── dataset/
│ └── Mental Health Dataset.csv
|
├── source file/
| └── Mental Health Dataset.ipynb

##-----------------------------------------------------------------------------------

# 🚀 How to Run

# 1. Clone repo
## bash

git clone https://github.com/sirUttam/mental-health-ml-project.git

cd mental-health-ml-project

##-----------------------------------------------------------------------------------

Install dependencies
pip install -r requirements.txt

##-----------------------------------------------------------------------------------

Run app
streamlit run app.py

##------------------------------------------------------------------------------------

⚠️ Important Note

This project is for educational and analytical purposes only.
It is not a medical diagnosis tool.

##-----------------------------------------------------------------------------------

👨‍💻 Author

Built by: Uttam Aryal
Focus: Data Science | Machine Learning | Real-world Projects

##-----------------------------------------------------------------------------------
