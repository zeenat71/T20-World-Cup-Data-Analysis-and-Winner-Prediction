# 🏏 T20 World Cup Data Analysis and Winner Prediction

This project is a complete **Streamlit-based data analytics and machine learning application** developed to analyze ICC T20 World Cup match data and predict match winners based on given inputs.

---

## 🔍 Objective

* Analyze the T20 World Cup dataset to extract patterns and trends.
* Build a machine learning model to predict the winning team based on input features.
* Provide interactive visualizations and prediction tools using Streamlit.

---

## 📂 Dataset Description

The dataset used contains match-level information from ICC T20 World Cup tournaments.

### ✅ Columns Used:

* **Year** – Year of the match
* **Team 1** – First team playing
* **Team 2** – Second team playing
* **Winner** – Match winner
* **WC winner** – Indicates the team that won the World Cup that year
* **Venue** – Match location
* **Match Type** – Group Stage, Semi Final, Final, etc.
* **Host Nation** – Country hosting the match

---

## 🪑 Step-by-Step Breakdown

### 1. 📁 Upload & Load Dataset

* Users upload a CSV file using Streamlit file uploader.
* The file is read using `pandas.read_csv()`.

### 2. 🧹 Data Cleaning

* Missing values handled with:

  * Mode for categorical columns
  * Median for numerical (if any)
* Uniformity ensured by stripping spaces, replacing null-like values (`"None"`, empty, etc.)

### 3. 📊 Exploratory Data Analysis (EDA)

#### 🎈 Visualizations Implemented:

* **Pie Chart** – Proportion of wins by different teams
* **Bar Chart** – Most used venues
* **Count Plot** – Match winners by year
* **Swarm Plot** – Match type variation per year
* **Radar Plot** – Most common match types for top 3 Team 1s
* **Line Plot** – Yearly winning trends
* **Heatmap** – Team 1 vs Team 2 matchup frequency
* **Point Plot** – Number of unique venues each year
* **Donut Chart** – Match distribution across host nations

These visualizations are built using:

* `matplotlib`
* `seaborn`
* `plotly`

### 4. ⚖️ Correlation & Outlier Analysis

* **Label Encoding** temporarily applied to categorical features
* **Correlation Matrix (Heatmap)** shows how features are related
* **Boxplots and Swarmplots** used for outlier detection

### 5. 🚀 Machine Learning Model

#### 🔎 Problem:

* **Classification Task** – Predict match winner (target: `Winner`)

#### ⚛️ Features Selected:

* `Team 1`, `Team 2`, `Venue`, `Host Nation`, `Match Type`

#### 🔧 Preprocessing:

* Used `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
* Avoided deprecated `sparse` argument (replaced with `sparse_output`)

#### 🔄 Pipeline:

* Built using `Pipeline()` from sklearn combining encoding + model training

#### 🏋️ Model Used:

* `HistGradientBoostingClassifier` was selected because it handles categorical features efficiently without needing extensive preprocessing, is robust to overfitting, and performs well even with moderate-size datasets.
* This model is also less sensitive to missing values and works well with encoded categorical data.

#### ⚖️ Train-Test Split:

* 80/20 split using `train_test_split` with `stratify=y` for balanced class distribution
* Exception handling added for cases where some classes have too few examples

#### ⚖️ Hyperparameter Tuning:

* Done with `GridSearchCV`
* Parameters tuned: `max_depth`, `min_samples_leaf`, `n_estimators`, `min_samples_split`

#### 🔢 Evaluation Metrics:

* **Accuracy Score**
* **Classification Report** (Precision, Recall, F1)

---

## 🧠 Live Prediction Feature

* User selects input values from dropdowns:

  * `Team 1`, `Team 2`, `Venue`, `Match Type`, `Host Nation`
* Model predicts and shows the winning team instantly on screen

---

## 🚀 Tech Stack

* **Language**: Python
* **Libraries**:

  * `pandas`, `numpy`
  * `matplotlib`, `seaborn`, `plotly`
  * `sklearn`
  * `streamlit`

---

## 🚪 How to Run

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

3. Upload the dataset and explore the app

---

## 📅 Future Scope

* Add player-level stats
* Integrate live data APIs
* Improve accuracy with ensemble stacking
* Deploy on cloud (Heroku, Streamlit Cloud, etc.)

---

## 📄 Sample Dataset Format

| Year | Team 1 | Team 2 | Winner | WC winner | Venue | Match Type  | Host Nation |
| ---- | ------ | ------ | ------ | --------- | ----- | ----------- | ----------- |
| 2021 | India  | NZ     | NZ     | AUS       | Dubai | Group Stage | UAE         |
