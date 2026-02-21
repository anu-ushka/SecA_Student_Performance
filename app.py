# ==============================
# 📊 STUDENT LEARNING ANALYTICS
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix


# ------------------------------
# PAGE TITLE
# ------------------------------
st.set_page_config(page_title="Student AI Dashboard", layout="wide")

st.title("📊 Student Academic Performance Intelligence Dashboard")
st.markdown("### Predictive Insights Powered by Machine Learning")


# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # BASIC CHECK
    # ------------------------------
    required_columns = ['quiz_score', 'assignment_score', 'time_spent', 'final_score']

    if not all(col in df.columns for col in required_columns):
        st.error("CSV must contain: quiz_score, assignment_score, time_spent, final_score")
    else:

        # ------------------------------
        # PREPROCESSING
        # ------------------------------
        df = df.dropna()

        X = df[['quiz_score', 'assignment_score', 'time_spent']]
        y_reg = df['final_score']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # =====================================================
        # 1️⃣ LINEAR REGRESSION (Predict Final Score)
        # =====================================================
        st.header("📈 Linear Regression - Score Prediction")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_reg, test_size=0.2, random_state=42
        )

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2, col3 = st.columns(3)

        col1.metric("🎯 Avg Predicted Score", f"{np.mean(y_pred):.2f}")
        col2.metric("📈 R² Score", f"{r2:.3f}")
        col3.metric("📉 RMSE", f"{rmse:.2f}")

        # Graph
        st.subheader("Actual vs Predicted Scores")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred)
        ax1.set_xlabel("Actual Score")
        ax1.set_ylabel("Predicted Score")
        st.pyplot(fig1)


        # =====================================================
        # 2️⃣ LOGISTIC REGRESSION (Pass / Fail)
        # =====================================================
        st.header("🎓 Logistic Regression - Pass/Fail Classification")

        df['pass_fail'] = df['final_score'].apply(lambda x: 1 if x >= 40 else 0)

        y_class = df['pass_fail']

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_class, test_size=0.2, random_state=42
        )

        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)

        y_pred_class = log_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred_class)

        st.metric("🎯 Classification Accuracy", f"{acc:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_class)

        st.subheader("Confusion Matrix")
        fig2, ax2 = plt.subplots()
        ax2.imshow(cm)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)


        # =====================================================
        # 3️⃣ K-MEANS CLUSTERING (Learner Groups)
        # =====================================================
        st.header("🧠 K-Means Clustering - Learner Segmentation")

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        st.write("Cluster Distribution")
        st.write(df['Cluster'].value_counts())

        # Cluster Visualization
        st.subheader("Cluster Visualization (Quiz vs Assignment)")
        fig3, ax3 = plt.subplots()
        scatter = ax3.scatter(
            df['quiz_score'],
            df['assignment_score'],
            c=clusters
        )
        ax3.set_xlabel("Quiz Score")
        ax3.set_ylabel("Assignment Score")
        st.pyplot(fig3)

        # =====================================================
        # BASIC RECOMMENDATION LOGIC
        # =====================================================
        st.header("📌 Basic Study Recommendations")

        cluster_means = df.groupby('Cluster')[['quiz_score', 'assignment_score']].mean()

        for cluster in cluster_means.index:
            st.write(f"### Cluster {cluster}")
            if cluster_means.loc[cluster, 'quiz_score'] < 50:
                st.write("🔹 Needs improvement in quizzes.")
            if cluster_means.loc[cluster, 'assignment_score'] < 50:
                st.write("🔹 Needs improvement in assignments.")
            else:
                st.write("🔹 Performing well. Focus on advanced practice.")

else:
    st.info("Upload a CSV file to begin.")