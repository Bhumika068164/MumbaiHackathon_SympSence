# Symsence – AI Disease Predictor

**Symsence** is an AI-powered web application that predicts possible diseases based on user-inputted symptoms. Developed for **Mumbai Hacks 2025**, this project helps users get quick and reliable insights into their health through a user-friendly interface.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4.[Project Structure](#project-structure)  
5. [Installation](#installation)  
6. [Usage](#usage)   
7. [Contributing](#contributing)  
 

---

## Project Overview
Symsence allows users to input their symptoms and get predicted diseases instantly. The application uses machine learning models trained on a dataset of diseases and symptoms, providing accurate and fast predictions. This project emphasizes accessibility and ease of use for non-technical users.

---

## Features
- Input multiple symptoms easily  
- Instant disease prediction  
- User-friendly and intuitive interface  
- Backend powered by AI/ML model  
- Optional: feature for augmented datasets for improved predictions  

---

## Tech Stack
- **Backend:** Python, Flask  
- **Data Handling & ML:** Pandas, NumPy, Scikit-learn  
- **Frontend:** HTML, CSS, JavaScript  
- **Machine Learning Techniques:** Data preprocessing, model training, disease prediction algorithms

## Project Structure  

---Symsence/
│
├── dataset/                 # Original dataset files
├── augmented_dataset/       # Augmented datasets for improved model training
├── model/                   # Trained ML models
├── app.py                   # Main Flask application
├── requirements.txt         # Python dependencies
├── templates/               # HTML templates
├── static/                  # CSS, JS, images
├── data_preprocessing.py    # Code for dataset cleaning and preprocessing
├── model_training.py        # Code to train ML model
└── README.md

## Installation
Follow these steps to run the project locally:

1. **Clone the repository:**  
   
   git clone https://github.com/Bhumika068164/MumbaiHackathon_SympSence.git
   cd MumbaiHackathon_SympSence
2. Create a virtual environment:
python -m venv venv
3. Activate the virtual environment:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

4. Install required dependencies:


pip install -r requirements.txt

5. Run the Flask app:

python app.py

6. Open your browser and go to:


http://127.0.0.1:5000

##Usage

1. Enter your symptoms in the input form.

2. Click the Predict button.

3. View the predicted disease(s) instantly.

##Contributing

1. Contributions are welcome! If you want to improve Symsence, please follow these steps:

2. Fork the repository

3. Create a new branch (git checkout -b feature-name)

4. Make your changes

5. Commit (git commit -m 'Add some feature')

6. Push (git push origin feature-name)

7. Create a Pull Request




