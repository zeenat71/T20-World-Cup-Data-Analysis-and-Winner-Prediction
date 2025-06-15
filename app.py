import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier


# App config
st.set_page_config(page_title="🏏 T20 World Cup Analysis", page_icon="🏆", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.family'] = 'Segoe UI Emoji'

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Introduction", "Upload & Overview", "EDA & Visuals", "Correlation & Outliers", "ML Model & Prediction", "Conclusion"])

def load_worldcup():
    st.sidebar.subheader("Upload World Cup Data")
    f = st.sidebar.file_uploader("CSV file", type="csv")
    if f:
        return pd.read_csv(f)
    st.warning("Please upload dataset.")
    return None

def project_intro():
    st.markdown("""
    # 🏏 T20 World Cup Data Analysis Dashboard

    Welcome to the **T20 World Cup Analysis App**!  
    This interactive dashboard lets you explore key insights from T20 World Cup cricket tournaments.

    ### 📌 Key Features:
    - 📂 Upload your World Cup dataset
    - 📊 View insightful graphs and patterns from the data
    - 🔍 Detect correlations and outliers
    - 🤖 Run a machine learning model to predict match results
    - 📈 Explore team stats, venues, match types, and more!

    Whether you're a **data science enthusiast** or a **cricket fan**, this app is designed to combine analytics with sports excitement. 🏆

    ---
    """)


def clean_missing_values(df):
    # Convert placeholders to np.nan
    df.replace(["", " ", "NaN", "nan", "None", None], np.nan, inplace=True)

    # Fill Winner specifically
    if 'Winner' in df.columns and df['Winner'].isnull().sum() > 0:
        df['Winner'] = df['Winner'].fillna(df['Winner'].mode()[0])

    # Fill categorical missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fill numeric missing values
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())

    return df

def eda_visuals(df):
    st.header("General Stats")

    st.subheader("✅ Data Summary")
    st.write("🔢 Unique winners:", df['Winner'].nunique())
    st.write(df['Winner'].value_counts().head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Top Winners Pie Chart")
        top = df['Winner'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(top, labels=top.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

    with col2:
        st.write("### Host Nation vs Winner")
        ct = pd.crosstab(df['Host Nation'], df['Winner'])
        fig, ax = plt.subplots(figsize=(8,5))
        ct.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
        ax.set_ylabel("Matches Won")
        st.pyplot(fig)

    st.write("### Year-wise Winner Count")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(data=df, x='Year', hue='Winner', palette='Spectral', ax=ax)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.write("### Heatmap: Team 1 vs Team 2")
    tm = pd.crosstab(df['Team 1'], df['Team 2'])
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(tm, cmap='mako', ax=ax)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.write("### Donut Chart: Host Nation Frequency")
    hc = df['Host Nation'].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(hc, labels=hc.index, startangle=140, wedgeprops=dict(width=0.4), autopct='%1.1f%%')
    st.pyplot(fig)

    st.write("### Matches Per Year")
    fig, ax = plt.subplots(figsize=(10,5))
    df['Year'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    st.write("### Venues Diversity Per Year")
    vd = df.groupby('Year')['Venue'].nunique().reset_index()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.pointplot(data=vd, x='Year', y='Venue', color='orange', ax=ax)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.write("### Radar Chart: Top 3 Team 1 by Match Type")
    top3 = df['Team 1'].value_counts().head(3).index.tolist()
    mtypes = df['Match Type'].unique()
    data = [[df[(df['Team 1']==t)&(df['Match Type']==m)].shape[0] for m in mtypes] for t in top3]
    angles = np.linspace(0,2*np.pi,len(mtypes),endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    for i, d in enumerate(data):
        d += d[:1]
        ax.plot(angles, d, label=top3[i]); ax.fill(angles, d, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(mtypes)
    ax.legend(loc='upper right')
    st.pyplot(fig)

    st.write("### Team 2 Participation Frequency")
    team2_counts = df['Team 2'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=team2_counts.values, y=team2_counts.index, ax=ax, palette='cubehelix')
    ax.set_xlabel("Matches")
    ax.set_ylabel("Team 2")
    st.pyplot(fig)
    
    st.write("### Swarm Plot: Match Type Distribution by Year")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.swarmplot(data=df, x='Year', y='Match Type', ax=ax, palette='Set2')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    st.write("### Most Frequent Venues")
    venue_counts = df['Venue'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(y=venue_counts.index, x=venue_counts.values, ax=ax, palette='coolwarm')
    ax.set_xlabel("Match Count")
    ax.set_ylabel("Venue")
    st.pyplot(fig)
    
    st.write("### Winning Trends Over Years")
    win_trends = df.groupby(['Year','Winner']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12,6))
    win_trends.plot(ax=ax)
    ax.set_ylabel("Wins")
    ax.set_xlabel("Year")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    st.pyplot(fig)
    
    st.write("### Pairplot of Numeric Features")
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig = sns.pairplot(numeric_df)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for pairplot.")

    
def overview(df):
    st.header("Data Overview")
    st.write(df.info(), df.head())
    st.write("Missing values:\n", df.isnull().sum())
 




def corr_outliers(df):
    st.header("Correlation & Outlier Visualization")

    # Encode categorical columns for correlation
    enc = df.copy()
    for c in enc.select_dtypes(include='object').columns:
        enc[c] = LabelEncoder().fit_transform(enc[c].astype(str))

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(enc.corr(), cmap='coolwarm', annot=False, cbar=True, linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Boxplot + Swarmplot for Outlier Detection")

    numeric_cols = enc.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) == 0:
        st.info("No numeric columns found for outlier detection.")
        return

    # Visualize each numeric column with boxplot and swarmplot
    for col in numeric_cols:
        st.write(f"### Outliers in `{col}`")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=enc, x=col, color='lightblue', ax=ax)
        sns.swarmplot(data=enc, x=col, color='black', size=3, ax=ax)
        st.pyplot(fig)


def conclusion():
    st.header("Conclusion")
    st.write("""
    - Cleaned dataset: replaced missing winners with mode.
    - Visual insights include top winners, match-up patterns, 
      host effects, venue trends, and match types per team.
    - Correlation analysis and outlier detection ensure data quality.
    - Next steps: predictive modeling, advanced interactivity, 
      embedding maps or filters for deeper exploration.
    """)
    st.balloons()

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def ml_model_prediction(df):
    st.header("🤖 ML Model: Winner Prediction")

    df = df.dropna()
    features = ['Team 1', 'Team 2', 'Venue', 'Host Nation', 'Match Type']
    target = 'Winner'

    X = df[features]
    y = df[target]

    # Define the preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])

    # Split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Model Evaluation")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.markdown("---")
    st.subheader("🎯 Predict Winner (Runtime)")

    # Collect runtime user input
    input_data = {}
    for col in features:
        input_data[col] = st.selectbox(f"Select {col}", sorted(df[col].unique()))

    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    st.success(f"🏆 Predicted Winner: **{pred}**")


# Main flow
raw_df = load_worldcup()
df = clean_missing_values(raw_df.copy()) if raw_df is not None else None

if page == "Project Introduction":
    project_intro()

elif page == "Upload & Overview":
    if df is not None:
        overview(df)

elif page == "EDA & Visuals":
    if df is not None:
        eda_visuals(df)

elif page == "Correlation & Outliers":
    if df is not None:
        corr_outliers(df)

elif page == "ML Model & Prediction":
    if df is not None:
        ml_model_prediction(df)

elif page == "Conclusion":
    conclusion()

