# ==============================
# 📊 STUDENT AI PERFORMANCE DASHBOARD
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix

from preprocessing import preprocess_data


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Study Coach - Analytics", layout="wide")

st.title("🎓 Intelligent Learning Analytics Dashboard")
st.markdown("From Predictive ML to AI Study Coaching")


# -----------------------------
# LOAD DATA (AUTO)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Dataset/student_performance.csv", sep="\t")

df = load_data()

st.write("Shape of dataset:", df.shape)
st.write("Columns:", list(df.columns))
st.write(df.head())

# -----------------------------
# PREPROCESS USING YOUR PIPELINE
# -----------------------------
df, X_scaled, y_reg, scaler = preprocess_data(df)
st.write("Columns in dataset:", df.columns)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Score Prediction", "🎯 Pass/Fail", "🧠 Learner Segments", "📌 Recommendations"]
)


# ======================================================
# TAB 1 — LINEAR REGRESSION
# ======================================================
with tab1:

    st.subheader("Predicting Final Exam Scores")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    st.pyplot(fig)


# ======================================================
# TAB 2 — LOGISTIC REGRESSION
# ======================================================
with tab2:

    st.subheader("Pass / Fail Classification")

    y_class = df['Pass']   # Already created in preprocessing.py

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_class, test_size=0.2, random_state=42
    )

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_class)
    st.metric("Accuracy", f"{acc:.2f}")

    cm = confusion_matrix(y_test, y_pred_class)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# ======================================================
# TAB 3 — KMEANS CLUSTERING
# ======================================================
with tab3:

    st.subheader("Student Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    st.bar_chart(df['Cluster'].value_counts())

    fig, ax = plt.subplots()
    ax.scatter(
        df['StudyHours'],
        df['ExamScore'],
        c=clusters
    )
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Exam Score")
    st.pyplot(fig)


# ======================================================
# TAB 4 — SMART RECOMMENDATIONS
# ======================================================
with tab4:

    st.subheader("AI-Based Study Insights")

    cluster_means = df.groupby('Cluster')[['StudyHours', 'ExamScore']].mean()

    for cluster in cluster_means.index:

        st.markdown(f"### 🎓 Learner Group {cluster}")

        hours = cluster_means.loc[cluster, 'StudyHours']
        score = cluster_means.loc[cluster, 'ExamScore']

        if score < 50:
            st.warning("Needs improvement. Increase study consistency.")
        elif score >= 75:
            st.success("High Performer. Start advanced topics.")
        else:
            st.info("Moderate performer. Improve weak subject areas.")