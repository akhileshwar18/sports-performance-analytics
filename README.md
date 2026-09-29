# 🏅 Sports Performance Analytics & Athlete Segmentation

### 🤖 Machine Learning-Based Athlete Analysis, Performance Prediction & Segmentation

> A machine learning project that analyzes athlete performance data, groups athletes based on similar characteristics, predicts overall performance, estimates medal chance, and provides personalized training recommendations.

---

## 🌟 Project Overview

**Sports Performance Analytics & Athlete Segmentation** is a machine learning application designed to analyze different aspects of athlete performance.

The system uses athlete information such as:

* 🏃 Training experience
* ⏱️ Training hours
* 🫁 VO₂ Max
* ⚡ Sprint speed
* 🧠 Reaction time
* 💪 Strength
* ❤️ Resting heart rate
* 😴 Sleep
* 🔄 Recovery heart rate
* 📅 Training days

The system processes these inputs and provides:

**Athlete Data → K-Means Grouping → Random Forest Performance Prediction → Linear Regression Medal Chance → Recommendations**

---

## 🎯 Objectives

* Analyze athlete performance using machine learning.
* Group athletes with similar performance characteristics.
* Predict an athlete's overall performance score.
* Estimate the athlete's medal chance.
* Provide simple training recommendations.
* Present the results through an interactive web application.

---

## 🧠 Machine Learning Algorithms

### 1️⃣ K-Means Clustering

K-Means is used for **athlete segmentation**.

It groups athletes with similar performance characteristics into different groups.

Example:

```text
High Performer
Intermediate
Developing
```

The best number of clusters is selected using the **Silhouette Score**.

---

### 2️⃣ Random Forest Regression 🌲

Random Forest is used to predict:

```text
Predicted Overall Performance %
```

The model uses multiple athlete features and combines the predictions from several decision trees.

---

### 3️⃣ Linear Regression 📈

Linear Regression is used to estimate:

```text
Predicted Medal Chance %
```

It uses selected athlete information together with the predicted performance score.

---

## 🔄 Project Workflow

```text
              Athlete Dataset
                    │
                    ▼
             Data Preprocessing
                    │
                    ▼
              Feature Scaling
                    │
                    ▼
             K-Means Clustering
                    │
                    ▼
            Athlete Segmentation
                    │
                    ▼
          Random Forest Regression
                    │
                    ▼
       Overall Performance Prediction
                    │
                    ▼
           Linear Regression
                    │
                    ▼
          Medal Chance Prediction
                    │
                    ▼
        Personalized Recommendations
```

---

## 📊 Input Features

The application uses the following athlete features:

| Feature             | Description                  |
| ------------------- | ---------------------------- |
| Age                 | Athlete's age                |
| Weight              | Athlete's weight             |
| Training Experience | Years of training experience |
| Training Hours      | Weekly training hours        |
| VO₂ Max             | Aerobic fitness measurement  |
| Sprint Speed        | Sprint performance           |
| Reaction Time       | Reaction speed               |
| Strength Score      | Strength test score          |
| Resting Heart Rate  | Heart rate at rest           |
| Sleep Hours         | Daily sleep duration         |
| Training Days       | Training days per week       |
| Recovery Heart Rate | Recovery heart rate          |

---

## 💻 Application Features

### 🎯 Athlete Prediction

Enter athlete information through the interactive interface.

The application provides:

* 📊 Overall performance prediction
* 🏅 Performance group
* 🥇 Predicted medal chance
* 💡 Training recommendations

---

### 📈 Model Insights

The application also displays:

* Number of athletes in the dataset
* Random Forest R²
* Linear Regression R²
* Random Forest feature importance
* K-Means athlete group distribution

---

## 🖥️ Technology Stack

| Technology                   | Purpose                    |
| ---------------------------- | -------------------------- |
| 🐍 Python                    | Programming                |
| 🐼 Pandas                    | Data processing            |
| 🔢 NumPy                     | Numerical operations       |
| 🤖 Scikit-learn              | Machine learning           |
| 📊 Plotly                    | Interactive visualizations |
| 🎨 Streamlit                 | Web application            |
| 📁 Pickle                    | Saving trained models      |
| ☁️ Streamlit Community Cloud | Public deployment          |

---

## 📂 Project Structure

```text
sports-performance-analytics/
│
├── app.py
├── train_model.py
├── models.pkl
├── Sports_analytics_file.csv
├── requirements.txt
└── README.md
```

### File Description

**`app.py`**
Main Streamlit application that provides the user interface and predictions.

**`train_model.py`**
Trains the K-Means, Random Forest, and Linear Regression models.

**`models.pkl`**
Stores the trained models and required preprocessing information.

**`Sports_analytics_file.csv`**
Dataset used for model training.

**`requirements.txt`**
Contains the Python libraries required to run the application.

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/sports-performance-analytics.git
```

### 2. Open the project folder

```bash
cd sports-performance-analytics
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local address similar to:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application can be deployed publicly using **Streamlit Community Cloud**.

### Deployment Flow

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Select app.py
       ↓
Install requirements
       ↓
Deploy
       ↓
🌐 Public Web Application
```

---

## 📌 Example Prediction Flow

```text
New Athlete
     │
     ├── Athlete ID
     ├── Age
     ├── Weight
     ├── Training Experience
     ├── Training Hours
     ├── VO₂ Max
     ├── Sprint Speed
     ├── Reaction Time
     ├── Strength
     ├── Sleep
     └── Recovery Data
             │
             ▼
       Machine Learning
             │
       ┌─────┴─────┐
       ▼           ▼
   Athlete      Performance
   Group        Prediction
                     │
                     ▼
              Medal Chance
                     │
                     ▼
             Recommendations
```

---

## 💡 Recommendations System

The application compares selected athlete measurements with dataset reference values.

If predicted performance is **75% or above**, the application displays:

> **Good performance – maintain current training.**

For lower predicted performance, the system can provide recommendations related to areas such as:

* Training consistency
* Aerobic fitness
* Strength training
* Speed training
* Reaction-time practice
* Sleep and recovery
* Training frequency
* Cardiovascular fitness

---

## 🔮 Future Enhancements

Possible future improvements include:

* 📱 Mobile-friendly application
* 📊 Larger real-world athlete datasets
* 🧠 Additional machine learning algorithms
* 📈 Athlete performance tracking over time
* 🏋️ Personalized training plans
* 📊 Advanced athlete dashboards
* 🔐 User authentication
* ☁️ Cloud database integration
* 📅 Historical performance comparison

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**.

The included dataset is a synthetic/demo dataset and the predictions should not be treated as real-world athlete selection or professional sports decisions.

---

## 👨‍💻 Project

### 🏅 Sports Performance Analytics & Athlete Segmentation

**Machine Learning | Python | Streamlit | Data Analytics**

---

### ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub!
