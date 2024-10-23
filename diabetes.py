import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import itertools
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Choose a section", 
                               ["Dataset Overview", "Data Exploration", "Model Building", "Model Evaluation", "Conclusion"])

# Title
st.title("Diabetes Prediction and Analysis")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('diabetes.csv')

dataset = load_data()

# Dataset Overview Section
if options == "Dataset Overview":
    st.subheader("Dataset Overview")

    st.write("This dataset contains information on patients' medical details. The objective is to predict if a patient is likely to have diabetes or not.")
    
    # Dataset description and details about each feature
    st.write("""    
    ### Dataset Features:
    - **Pregnancies**: Number of times the patient has been pregnant.
    - **Glucose**: Plasma glucose concentration (mg/dL).
    - **Blood Pressure**: Diastolic blood pressure (mm Hg).
    - **SkinThickness**: Thickness of skin fold (mm).
    - **Insulin**: 2-hour serum insulin (mu U/ml).
    - **BMI**: Body Mass Index (weight in kg/(height in m)^2).
    - **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history.
    - **Age**: Age of the patient.
    - **Outcome**: Target variable (1 indicates diabetes, 0 indicates no diabetes).
    """)

    st.write(f"### Dataset Dimensions: {dataset.shape}")
    st.write(f"### Target Distribution: {dataset['Outcome'].value_counts()}")

    st.write("### First 10 Records of the Dataset")
    st.write(dataset.head(10))

    st.write("### Dataset Description")
    st.write(dataset.describe())

# Data Preprocessing - Move preprocessing to be available globally
@st.cache_data
def preprocess_data(dataset):
    # Replacing zero values with NaN
    dataset_new = dataset.copy()
    dataset_new[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset_new[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

    # Replace NaN values with mean
    dataset_new.fillna(dataset_new.mean(), inplace=True)

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset_new), columns=dataset.columns)

    # Feature and Target Selection
    X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values  # Selecting Glucose, Insulin, BMI, Age
    Y = dataset_scaled['Outcome'].values

    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=dataset_new['Outcome'])
    
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = preprocess_data(dataset)

# Sidebar settings
if options == "Data Exploration":
    st.sidebar.subheader("Data Exploration Options")
    show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", True)
    show_pairplot = st.sidebar.checkbox("Show Pairplot", True)

    # Data Exploration Section
    st.subheader("Data Exploration")

    # Preview data
    st.write(dataset.head(10))

    # Dataset Info
    with st.expander("Dataset Info"):
        st.write(f"Dataset Dimensions: {dataset.shape}")
        st.write(dataset.describe())
        st.write(dataset.info())

    # Count of missing values
    st.write("Missing Values in Dataset:")
    st.write(dataset.isnull().sum())

    # Visualizations
    st.subheader("Visualizations")

    # Countplot
    st.write("Outcome Count Plot:")
    sns.countplot(x='Outcome', data=dataset)
    st.pyplot(plt)

    # Feature Histograms
    st.write("Feature Distributions:")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    col = dataset.columns[:8]
    for i, j in itertools.zip_longest(col, range(len(col))):
        ax = axes[j//3, j%3]
        dataset[i].hist(bins=20, ax=ax)
        ax.set_title(i)
    st.pyplot(fig)

    # Correlation Heatmap (Optional)
    if show_heatmap:
        st.write("Correlation Heatmap:")
        sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # Pairplot (Optional)
    if show_pairplot:
        st.write("Generating Pairplot, please wait...")
        fig = sns.pairplot(data=dataset, hue='Outcome')
        st.pyplot(fig)  # Display the pairplot
        plt.close()  # Clear the plot to avoid overlap with other plots

elif options == "Model Building":
    # Model Building Section
    st.subheader("Model Building")

    st.write("## Data Preprocessing Completed!")
    st.write("The dataset has been preprocessed, and features have been scaled.")

    # Show Shapes
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")
    st.write(f"Y_train shape: {Y_train.shape}")
    st.write(f"Y_test shape: {Y_test.shape}")

elif options == "Model Evaluation":
    # Model Evaluation Section
    st.subheader("Model Evaluation")
    
    st.sidebar.subheader("Model Options")
    selected_model = st.sidebar.selectbox("Choose a model", 
                                          ["K-Nearest Neighbors", "Support Vector Classifier", "Naive Bayes", "Decision Tree"])
    
    # KNN: Slider for K value
    if selected_model == "K-Nearest Neighbors":
        k_value = st.sidebar.slider("Choose K (n_neighbors)", min_value=1, max_value=30, value=24)

        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)

        st.write(f"KNN Model Accuracy: {accuracy_score(Y_test, Y_pred) * 100:.2f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(Y_test, Y_pred)
        st.write("Confusion Matrix:")
        sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues")
        st.pyplot(plt)

        # Classification Report
        st.write("Classification Report:")
        report = classification_report(Y_test, Y_pred)
        st.text(report)

    # Support Vector Classifier
    if selected_model == "Support Vector Classifier":
        svc = SVC(kernel='linear', random_state=42)
        svc.fit(X_train, Y_train)
        Y_pred = svc.predict(X_test)
        
        st.write(f"SVC Model Accuracy: {accuracy_score(Y_test, Y_pred) * 100:.2f}%")

        # Confusion Matrix for SVC
        cm_svc = confusion_matrix(Y_test, Y_pred)
        st.write("SVC Confusion Matrix:")
        sns.heatmap(pd.DataFrame(cm_svc), annot=True, cmap="Blues")
        st.pyplot(plt)

        # Classification Report
        st.write("Classification Report:")
        report_svc = classification_report(Y_test, Y_pred)
        st.text(report_svc)

    # Naive Bayes
    if selected_model == "Naive Bayes":
        nb = GaussianNB()
        nb.fit(X_train, Y_train)
        Y_pred = nb.predict(X_test)

        st.write(f"Naive Bayes Accuracy: {accuracy_score(Y_test, Y_pred) * 100:.2f}%")

        # Confusion Matrix for Naive Bayes
        cm_nb = confusion_matrix(Y_test, Y_pred)
        st.write("Naive Bayes Confusion Matrix:")
        sns.heatmap(pd.DataFrame(cm_nb), annot=True, cmap="Blues")
        st.pyplot(plt)

        # Classification Report
        st.write("Classification Report:")
        report_nb = classification_report(Y_test, Y_pred)
        st.text(report_nb)

    # Decision Tree
    if selected_model == "Decision Tree":
        dtree = DecisionTreeClassifier(criterion='entropy', random_state=42)
        dtree.fit(X_train, Y_train)
        Y_pred = dtree.predict(X_test)

        st.write(f"Decision Tree Accuracy: {accuracy_score(Y_test, Y_pred) * 100:.2f}%")

        # Confusion Matrix for Decision Tree
        cm_dt = confusion_matrix(Y_test, Y_pred)
        st.write("Decision Tree Confusion Matrix:")
        sns.heatmap(pd.DataFrame(cm_dt), annot=True, cmap="Blues")
        st.pyplot(plt)

        # Classification Report
        st.write("Classification Report:")
        report_dt = classification_report(Y_test, Y_pred)
        st.text(report_dt)

elif options == "Conclusion":
    st.subheader("Conclusion")
    st.write("After evaluating all models, you can decide on the best one.")
    st.write("Based on accuracy, confusion matrix, and classification reports, the optimal model for predicting diabetes can be selected.")
    st.write("### Highest Accuracy: 78.57%")
    st.write("### K-Nearest Neighbors Accuracy: 78.57%")

# Set theme
st.markdown("""
<style>
    .css-1d391kg {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    .css-1y4p1p4 {
        background-color: #1c1f25 !important;
    }
</style>
""", unsafe_allow_html=True)
