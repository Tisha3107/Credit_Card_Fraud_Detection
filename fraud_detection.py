#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import io
import base64


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


# Load sample dataset
def load_sample_data():
    # Create a simplified sample dataset with similar structure
    np.random.seed(42)
    n_samples = 1000
    n_features = 28
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate some anomalies (0.5% fraud)
    n_frauds = int(0.005 * n_samples)
    
    # For fraudulent transactions, modify some features to be outliers
    for i in range(n_frauds):
        X[i, np.random.randint(0, n_features, 3)] += np.random.randn(3) * 5
    
    # Create class labels (0 for normal, 1 for fraud)
    y = np.zeros(n_samples)
    y[:n_frauds] = 1
    
    # Create a DataFrame with similar structure to creditcard.csv
    columns = ['V' + str(i+1) for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    # Add Time, Amount, and Class columns
    df['Time'] = np.random.randint(0, 172800, n_samples)  # Seconds in 2 days
    df['Amount'] = np.abs(np.random.randn(n_samples) * 100 + 50)  # Transaction amounts
    df['Class'] = y
    
    # Make fraudulent transactions have slightly higher amounts
    df.loc[df['Class'] == 1, 'Amount'] *= 1.5
    
    return df


# In[4]:


# 2. Data Preprocessing
def preprocess_data(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # Check the distribution of the 'Class' variable
    class_distribution = df['Class'].value_counts()
    
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
    
    return X_scaled, df['Class'], missing_values, class_distribution


# In[5]:


# 3. Dimensionality Reduction with PCA for Visualization
def apply_pca(X_scaled):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    explained_variance = pca.explained_variance_ratio_
    return pca_df, pca, explained_variance


# In[6]:


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


# In[7]:


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


# In[8]:


# 4.2 Local Outlier Factor
def local_outlier_factor(X_scaled, contamination=0.01):
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    # Prediction: 1 for inliers, -1 for outliers
    y_pred = model.fit_predict(X_scaled)
    # Convert to binary: 0 for inliers, 1 for outliers (frauds)
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    return y_pred, model


# In[9]:


# 4.3 DBSCAN
def dbscan_model(X_scaled, eps=0.3, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X_scaled)
    # In DBSCAN, -1 represents outliers
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    return y_pred, model


# In[10]:


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


# In[11]:


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


# In[12]:


# Generate a downloadable report
def generate_report(df, X_scaled, y_true, y_pred, model_choice, results, missing_values, class_distribution):
    report = io.StringIO()
    
    report.write("# Credit Card Fraud Detection Report\n\n")
    report.write(f"## Dataset Overview\n")
    report.write(f"- Total transactions: {len(df)}\n")
    report.write(f"- Normal transactions: {sum(y_true == 0)}\n")
    report.write(f"- Fraudulent transactions: {sum(y_true == 1)}\n")
    report.write(f"- Fraud percentage: {sum(y_true == 1)/len(y_true)*100:.2f}%\n\n")
    
    report.write(f"## Data Quality\n")
    report.write(f"- Missing values: {sum(missing_values)}\n\n")
    
    report.write(f"## Model: {model_choice}\n")
    report.write(f"- Accuracy: {results['Accuracy']:.4f}\n")
    report.write(f"- Precision: {results['Precision']:.4f}\n")
    report.write(f"- Recall: {results['Recall']:.4f}\n")
    report.write(f"- F1 Score: {results['F1']:.4f}\n\n")
    
    report.write(f"## Prediction Results\n")
    report.write(f"- True Positives: {sum((y_true == 1) & (np.array(y_pred) == 1))}\n")
    report.write(f"- False Positives: {sum((y_true == 0) & (np.array(y_pred) == 1))}\n")
    report.write(f"- True Negatives: {sum((y_true == 0) & (np.array(y_pred) == 0))}\n")
    report.write(f"- False Negatives: {sum((y_true == 1) & (np.array(y_pred) == 0))}\n\n")
    
    return report.getvalue()


# In[13]:


# 7. Streamlit Application
def create_streamlit_app():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    
    st.title("Credit Card Fraud Detection")
    st.markdown("""
    This application uses unsupervised machine learning to detect fraudulent credit card transactions.
    You can upload your own dataset or use a sample dataset to get started.
    """)
    
    # Sidebar for data selection options
    st.sidebar.title("Data Options")
    data_option = st.sidebar.radio(
        "Select data source",
        ["Upload your own dataset", "Use sample dataset"]
    )
    
    # Initialize df as None
    df = None
    
    # Handle data selection
    if data_option == "Upload your own dataset":
        # File uploader
        uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
    else:
        # Use sample data
        if st.button("Load Sample Dataset"):
            with st.spinner("Generating sample dataset..."):
                df = load_sample_data()
                st.success("Sample dataset loaded successfully!")
    
    # Only continue if we have data
    if df is not None:
        # Display raw data overview
        with st.expander("View Raw Data Preview"):
            st.dataframe(df.head())
            st.write(f"Dataset Shape: {df.shape}")
        
        # Data preprocessing
        X_scaled, y_true, missing_values, class_distribution = preprocess_data(df)
        
        # Show data statistics
        with st.expander("Data Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Feature Statistics:")
                st.dataframe(X_scaled.describe())
            
            with col2:
                # Class distribution chart
                fig = px.pie(names=['Normal', 'Fraud'], 
                             values=df['Class'].value_counts().values, 
                             title='Transaction Class Distribution')
                st.plotly_chart(fig)
                
                st.info("""
                **What does this chart mean?**
                
                This pie chart shows the distribution of normal vs. fraudulent transactions in the dataset.
                Credit card fraud datasets are typically highly imbalanced, with fraudulent transactions
                representing a very small percentage of the total transactions.
                """)
        
        # Feature exploration
        with st.expander("Feature Exploration"):
            selected_features = st.multiselect(
                "Select features to explore",
                X_scaled.columns.tolist(),
                default=X_scaled.columns[:2].tolist()
            )
            
            if selected_features:
                # Create distributions by class
                fig = go.Figure()
                
                for feature in selected_features:
                    fig.add_trace(go.Violin(
                        x=['Normal']*sum(y_true == 0),
                        y=X_scaled[feature][y_true == 0],
                        name=f'{feature} - Normal',
                        box_visible=True,
                        meanline_visible=True,
                        side='negative',
                        line_color='blue'
                    ))
                    
                    fig.add_trace(go.Violin(
                        x=['Fraud']*sum(y_true == 1),
                        y=X_scaled[feature][y_true == 1],
                        name=f'{feature} - Fraud',
                        box_visible=True,
                        meanline_visible=True,
                        side='positive',
                        line_color='red'
                    ))
                
                fig.update_layout(
                    title="Distribution of Selected Features by Class",
                    xaxis_title="Transaction Class",
                    yaxis_title="Feature Value",
                    violinmode='overlay'
                )
                
                st.plotly_chart(fig)
                
                st.info("""
                **What does this chart mean?**
                
                These violin plots show the distribution of feature values for normal vs. fraudulent transactions.
                Significant differences in distributions indicate that the feature might be useful for detecting fraud.
                The wider sections of the violin plot represent a higher probability of observations taking that value.
                """)
        
        # Apply PCA
        pca_df, pca, explained_variance = apply_pca(X_scaled)
        
        # Model selection sidebar
        st.sidebar.title("Model Configuration")
        model_choice = st.sidebar.selectbox(
            "Select Unsupervised Learning Model",
            ["Isolation Forest", "Local Outlier Factor", "DBSCAN"]
        )
        
        # Add model explanations in the sidebar
        if model_choice == "Isolation Forest":
            st.sidebar.info("""
            **Isolation Forest** works by isolating observations by randomly selecting a feature 
            and then randomly selecting a split value between the maximum and minimum values of 
            the selected feature. Since isolating anomalies is easier (fewer splits needed), 
            the algorithm can identify them with shorter paths in the decision tree.
            """)
            contamination = st.sidebar.slider("Contamination (expected % of outliers)", 0.001, 0.1, 0.01, 0.001)
            
            with st.spinner('Training Isolation Forest model...'):
                y_pred, model = isolation_forest_model(X_scaled, contamination)
                
        elif model_choice == "Local Outlier Factor":
            st.sidebar.info("""
            **Local Outlier Factor (LOF)** measures the local deviation of density of a given sample 
            with respect to its neighbors. It detects outliers by finding data points that have a 
            substantially lower density than their neighbors.
            """)
            contamination = st.sidebar.slider("Contamination (expected % of outliers)", 0.001, 0.1, 0.01, 0.001)
            n_neighbors = st.sidebar.slider("Number of Neighbors", 5, 50, 20)
            
            with st.spinner('Training Local Outlier Factor model...'):
                model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                y_pred = model.fit_predict(X_scaled.values)
                y_pred = [1 if pred == -1 else 0 for pred in y_pred]
                
        elif model_choice == "DBSCAN":
            st.sidebar.info("""
            **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** groups 
            together points that are closely packed together, marking as outliers points that 
            lie alone in low-density regions. It's effective for data without a clear cluster structure.
            """)
            eps = st.sidebar.slider("Epsilon (neighborhood distance)", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.sidebar.slider("Min Samples (min points in neighborhood)", 5, 100, 10)
            
            with st.spinner('Training DBSCAN model...'):
                y_pred, model = dbscan_model(X_scaled, eps, min_samples)
        
        # Model evaluation
        results = evaluate_model(y_true, y_pred)
        
        # Display results
        st.header("Model Results")
        
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{results['Accuracy']:.4f}")
        col2.metric("Precision", f"{results['Precision']:.4f}")
        col3.metric("Recall", f"{results['Recall']:.4f}")
        col4.metric("F1 Score", f"{results['F1']:.4f}")  # Key is 'F1'

        # Add Fraud Detection Summary right after metrics
        st.subheader("📊 Fraud Detection Summary")

        # Create summary statistics
        total_transactions = len(y_true)
        actual_frauds = sum(y_true)
        predicted_frauds = sum(y_pred)
        correct_fraud_predictions = sum((y_true == 1) & (np.array(y_pred) == 1))
        missed_frauds = sum((y_true == 1) & (np.array(y_pred) == 0))
        false_alarms = sum((y_true == 0) & (np.array(y_pred) == 1))

        # Display summary statistics in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Transaction Analysis**:
            - Total Transactions: {total_transactions:,}
            - Actual Fraudulent Transactions: {actual_frauds:,} ({actual_frauds/total_transactions:.2%})
            - Predicted Fraudulent Transactions: {predicted_frauds:,} ({predicted_frauds/total_transactions:.2%})
            """)

        with col2:
            st.markdown(f"""
            **Fraud Detection Results**:
            - Correctly Identified Frauds: {correct_fraud_predictions:,} ({correct_fraud_predictions/actual_frauds:.2%} of all frauds)
            - Missed Frauds: {missed_frauds:,} ({missed_frauds/actual_frauds:.2%} of all frauds)
            - False Alarms: {false_alarms:,} ({false_alarms/predicted_frauds:.2%} of fraud predictions)
            """)

        # Add a visual indicator of overall effectiveness
        fraud_detection_rate = correct_fraud_predictions/actual_frauds if actual_frauds > 0 else 0
        effectiveness_color = "red" if fraud_detection_rate < 0.5 else "orange" if fraud_detection_rate < 0.8 else "green"

        st.markdown(f"""
        <div style='background-color: {effectiveness_color}; padding: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>Detection Effectiveness: {fraud_detection_rate:.2%}</h3>
            <p style='color: white; margin: 0;'>
                {
                "⚠️ POOR - The majority of frauds are being missed!" if fraud_detection_rate < 0.5 else
                "⚠️ MODERATE - A significant number of frauds are still being missed." if fraud_detection_rate < 0.8 else
                "✅ GOOD - Most fraudulent transactions are being detected."
                }
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("What do these metrics mean?"):
            st.markdown("""
            **Accuracy**: The proportion of correctly classified transactions (both normal and fraudulent).
    
            **Precision**: The proportion of predicted frauds that are actually fraudulent. Higher precision means fewer false alarms.
    
            **Recall**: The proportion of actual frauds that were correctly identified. Higher recall means fewer missed frauds.
    
            **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
    
            In fraud detection, recall is often more important than precision, as the cost of missing a fraudulent transaction (false negative) is typically higher than the cost of investigating a legitimate transaction (false positive).
            """)
        
        
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Normal', 'Predicted Fraud'],
                y=['Actual Normal', 'Actual Fraud'],
                hoverongaps=False,
                colorscale='Viridis',
                text=cm,
                texttemplate="%{text}"))
        
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig)
        
        with st.expander("What does this confusion matrix mean?"):
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            **True Negatives ({tn})**: Normal transactions correctly identified as normal.
            
            **False Positives ({fp})**: Normal transactions incorrectly flagged as fraudulent.
            
            **False Negatives ({fn})**: Fraudulent transactions missed by the model.
            
            **True Positives ({tp})**: Fraudulent transactions correctly identified as fraudulent.
            
            In fraud detection, minimizing False Negatives (missed frauds) is typically more important than minimizing False Positives (false alarms).
            """)
        
        # PCA Visualization
        st.subheader("PCA Visualization")
        
        # Explain PCA
        with st.expander("What is PCA and how to interpret this visualization?"):
            st.markdown(f"""
            **Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms the data into a new coordinate system.
            
            In this visualization:
            - Each point represents a transaction
            - The x-axis (PC1) and y-axis (PC2) are the two principal components that capture the most variance in the data
            - PC1 explains {explained_variance[0]*100:.2f}% of the variance
            - PC2 explains {explained_variance[1]*100:.2f}% of the variance
            - Together they explain {sum(explained_variance)*100:.2f}% of the total variance
            
            If fraudulent transactions form distinct clusters in this visualization, it suggests that the model can effectively separate them from normal transactions.
            """)
        
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
                
                with st.expander("How to interpret feature importance?"):
                    st.markdown("""
                    **Feature importance** indicates which features are most useful for detecting fraud. 
                    
                    Features with higher importance scores have a stronger influence on the model's decisions.
                    In fraud detection, these might represent transaction characteristics that frequently differ
                    between legitimate and fraudulent activity.
                    
                    Paying attention to these features can help in:
                    - Understanding the patterns of fraudulent activity
                    - Creating more targeted fraud detection rules
                    - Focusing investigation efforts on the most suspicious aspects of transactions
                    """)
                
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
            
            with st.expander(f"How {model_choice} detects outliers"):
                if model_choice == "Local Outlier Factor":
                    st.markdown("""
                    **Local Outlier Factor (LOF)** calculates the local density deviation of a data point with respect to its neighbors. 
                    
                    Points with significantly lower density than their neighbors are considered outliers (potential frauds).
                    
                    In this visualization, red points represent transactions that the model considers anomalous based on their
                    difference from surrounding transactions in the feature space.
                    """)
                else:  # DBSCAN
                    st.markdown("""
                    **DBSCAN** identifies clusters of transactions with similar characteristics. 
                    
                    Transactions that don't belong to any cluster (aren't similar enough to a sufficient number of other transactions)
                    are flagged as potential frauds.
                    
                    In this visualization, red points represent transactions that don't fit into any of the identified clusters
                    of normal transaction patterns.
                    """)
        
        # Generate detailed report
        st.header("Analysis Report")
        report_content = generate_report(df, X_scaled, y_true, y_pred, model_choice, results, missing_values, class_distribution)
        
        st.download_button(
            label="Download Full Analysis Report",
            data=report_content,
            file_name="fraud_detection_report.md",
            mime="text/markdown"
        )
        
        # Download predictions
        prediction_df = pd.DataFrame({
            'Transaction_ID': range(len(y_true)),
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        st.download_button(
            label="Download Predictions",
            data=prediction_df.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )


# In[14]:


# Run the main Streamlit app
if __name__ == "__main__":
    create_streamlit_app()


# In[ ]:




