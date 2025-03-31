#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split


# In[2]:


# Load the dataset
def load_data():
    try:
        # Attempt to load from file if available
        df = pd.read_csv('creditcard.csv')
    except:
        # If file not available, notify user
        st.error("Please upload the creditcard.csv file to continue")
        df = None
    return df


# In[3]:


# 2. Data Preprocessing
def preprocess_data(df):
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())
    
    # Check the distribution of the 'Class' variable
    print("\nClass distribution:")
    print(df['Class'].value_counts())
    
    # Separate features from target
    X = df.drop(['Class', 'Amount', 'Time'], axis=1)
    
    # Scale the Amount feature
    amount = df['Amount'].values.reshape(-1, 1)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(amount)
    
    # Scale the Time feature
    time_feature = df['Time'].values.reshape(-1, 1)
    df['Time_scaled'] = scaler.fit_transform(time_feature)
    
    # Prepare scaled dataset with all features
    X_scaled = df.drop(['Class', 'Amount', 'Time'], axis=1)
    X_scaled['Amount_scaled'] = df['Amount_scaled']
    X_scaled['Time_scaled'] = df['Time_scaled']
    
    return X_scaled, df['Class']


# In[4]:


# 3. Dimensionality Reduction with PCA for Visualization
def apply_pca(X_scaled):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    return pca_df, pca


# In[5]:


# 4. Unsupervised Learning Models
# 4.1 Isolation Forest
def isolation_forest_model(X_scaled, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)
    # Prediction: 1 for inliers, -1 for outliers
    y_pred = model.predict(X_scaled)
    # Convert to binary: 0 for inliers, 1 for outliers (frauds)
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    return y_pred, model


# In[6]:


# Custom function to calculate feature importance for Isolation Forest
def calculate_feature_importance(model, X):
    """
    Calculate feature importance for Isolation Forest
    
    Instead of relying on the feature_importances_ attribute,
    we'll calculate importance based on the decision path depths
    """
    # For each feature, we'll measure how much it contributes to anomaly detection
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Get the decision paths
    # We'll use decision_function as a proxy for importance
    anomaly_scores = -model.decision_function(X)  # Higher score = more anomalous
    
    # Calculate feature importance by correlation with anomaly scores
    importance = np.zeros(n_features)
    
    for i in range(n_features):
        # Calculate correlation between feature and anomaly score
        corr = np.corrcoef(X.iloc[:, i], anomaly_scores)[0, 1]
        importance[i] = np.abs(corr)  # Take absolute value of correlation
    
    # Normalize importance
    if np.sum(importance) > 0:
        importance = importance / np.sum(importance)
        
    return importance


# In[7]:


# 4.2 Local Outlier Factor
def local_outlier_factor(X_scaled, contamination=0.01):
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    # Prediction: 1 for inliers, -1 for outliers
    y_pred = model.fit_predict(X_scaled)
    # Convert to binary: 0 for inliers, 1 for outliers (frauds)
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    return y_pred, model


# In[8]:


# 4.3 DBSCAN
def dbscan_model(X_scaled, eps=0.3, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X_scaled)
    # In DBSCAN, -1 represents outliers
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    return y_pred, model


# In[9]:


# 5. Model Evaluation
def evaluate_model(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1  # Key is 'F1', not 'F1 Score'
    }


# In[10]:


# 6. Visualization Functions
def plot_pca_results(pca_df, y_true, y_pred=None):
    plt.figure(figsize=(12, 8))
    
    if y_pred is not None:
        # Plot with predicted labels
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=y_pred, cmap='coolwarm', alpha=0.7)
        plt.title('PCA of Credit Card Transactions with Predicted Fraud')
    else:
        # Plot with actual labels
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=y_true, cmap='coolwarm', alpha=0.7)
        plt.title('PCA of Credit Card Transactions with Actual Fraud')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.show()


# In[11]:


# 7. Streamlit Application
def create_streamlit_app():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    
    st.title("Credit Card Fraud Detection - Unsupervised Machine Learning")
    st.markdown("""
    This application uses unsupervised machine learning to detect fraudulent credit card transactions.
    Upload your creditcard.csv file to get started.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Display raw data overview
        with st.expander("View Raw Data Preview"):
            st.dataframe(df.head())
            st.write(f"Dataset Shape: {df.shape}")
        
        # Data preprocessing
        X_scaled, y_true = preprocess_data(df)
        
        # Show data statistics
        with st.expander("Data Statistics"):
            st.write("Feature Statistics:")
            st.dataframe(X_scaled.describe())
            
            # Class distribution chart
            fig = px.pie(names=['Normal', 'Fraud'], 
                         values=df['Class'].value_counts().values, 
                         title='Transaction Class Distribution')
            st.plotly_chart(fig)
        
        # Apply PCA
        pca_df, pca = apply_pca(X_scaled)
        
        # Model selection sidebar
        st.sidebar.title("Model Configuration")
        model_choice = st.sidebar.selectbox(
            "Select Unsupervised Learning Model",
            ["Isolation Forest", "Local Outlier Factor", "DBSCAN"]
        )
        
        # Model hyperparameters
        if model_choice == "Isolation Forest":
            contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01, 0.001)
            
            with st.spinner('Training Isolation Forest model...'):
                y_pred, model = isolation_forest_model(X_scaled, contamination)
                
        elif model_choice == "Local Outlier Factor":
            contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01, 0.001)
            n_neighbors = st.sidebar.slider("Number of Neighbors", 5, 50, 20)
            
            with st.spinner('Training Local Outlier Factor model...'):
                model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                y_pred = model.fit_predict(X_scaled.values)
                y_pred = [1 if pred == -1 else 0 for pred in y_pred]
                
        elif model_choice == "DBSCAN":
            eps = st.sidebar.slider("Epsilon", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.sidebar.slider("Min Samples", 5, 100, 10)
            
            with st.spinner('Training DBSCAN model...'):
                y_pred, model = dbscan_model(X_scaled, eps, min_samples)
        
        # Model evaluation
        results = evaluate_model(y_true, y_pred)
        
        # Display results
        st.header("Model Results")
        
        # Metrics - Fixed to use the correct keys
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{results['Accuracy']:.4f}")
        col2.metric("Precision", f"{results['Precision']:.4f}")
        col3.metric("Recall", f"{results['Recall']:.4f}")
        col4.metric("F1 Score", f"{results['F1']:.4f}")  # Key is 'F1'
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Normal', 'Predicted Fraud'],
                y=['Actual Normal', 'Actual Fraud'],
                hoverongaps=False,
                colorscale='Viridis'))
        
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig)
        
        # PCA Visualization
        st.subheader("PCA Visualization")
        
        # Create a dataframe with PCA components and class
        pca_viz_df = pd.DataFrame({
            'PC1': pca_df['PC1'],
            'PC2': pca_df['PC2'],
            'Actual Class': y_true,
            'Predicted Class': y_pred
        })
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Actual vs. Predicted", "Detailed View"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.scatter(pca_viz_df, x='PC1', y='PC2', color='Actual Class',
                              title='PCA with Actual Fraud Labels',
                              color_continuous_scale='Viridis')
                st.plotly_chart(fig1)
                
            with col2:
                fig2 = px.scatter(pca_viz_df, x='PC1', y='PC2', color='Predicted Class',
                              title='PCA with Predicted Fraud Labels',
                              color_continuous_scale='Viridis')
                st.plotly_chart(fig2)
        
        with tab2:
            # Interactive scatter plot
            st.subheader("Interactive PCA Plot")
            fig = px.scatter(
                pca_viz_df, x='PC1', y='PC2',
                color='Predicted Class',
                hover_data=['Actual Class'],
                title='PCA of Credit Card Transactions',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig)
        
        # Model-specific visualizations
        st.header(f"{model_choice} Analysis")
        
        if model_choice == "Isolation Forest":
            # Feature importance for Isolation Forest using our custom function
            st.subheader("Feature Importance")
            
            # Calculate feature importance using our custom function
            try:
                # Use our custom function to calculate feature importance
                importances = calculate_feature_importance(model, X_scaled)
                indices = np.argsort(importances)[::-1]
                features = X_scaled.columns
                
                fig = go.Figure(go.Bar(
                    x=[features[i] for i in indices],
                    y=importances[indices],
                    marker_color='green'
                ))
                fig.update_layout(title="Feature Importance in Isolation Forest",
                                 xaxis_title="Features",
                                 yaxis_title="Importance Score")
                st.plotly_chart(fig)
                
            except Exception as e:
                st.warning(f"Could not calculate feature importance. Details: {str(e)}")
                st.info("Showing feature distribution instead.")
                
                # Show feature distributions as an alternative
                fig = px.box(X_scaled, title="Feature Distributions")
                st.plotly_chart(fig)
            
        elif model_choice in ["Local Outlier Factor", "DBSCAN"]:
            # Show anomaly scores distribution
            st.subheader("Transaction Distribution")
            
            # Create a figure with outliers highlighted
            fig = px.scatter(
                x=range(len(y_pred)),
                y=X_scaled.iloc[:, 0],  # Using first feature as y-axis
                color=[('Fraud' if p == 1 else 'Normal') for p in y_pred],
                title=f"Transactions with {model_choice} Outliers Highlighted",
                labels={"x": "Transaction Index", "y": "Feature V1"},
                color_discrete_map={"Normal": "blue", "Fraud": "red"}
            )
            st.plotly_chart(fig)
        
        # Download predictions
        prediction_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        st.download_button(
            label="Download Predictions",
            data=prediction_df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )


# In[12]:


# Run the main Streamlit app
if __name__ == "__main__":
    create_streamlit_app()


# In[ ]:




