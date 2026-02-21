import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    
    df = df.drop_duplicates()

    df['Pass'] = (df['ExamScore'] >= 50).astype(int)

    df["StudyEfficiency"] = df["ExamScore"] / (df["StudyHours"] + 1)
    df["ConsistencyScore"] = (
        df["Attendance"] + df["AssignmentCompletion"]
    ) / 2

    features = [
        'StudyHours','Attendance','Resources','Motivation',
        'OnlineCourses','Discussions','AssignmentCompletion',
        'Age','StressLevel','StudyEfficiency','ConsistencyScore'
    ]

    X = df[features]
    y = df['ExamScore']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, scaler