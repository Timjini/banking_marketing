import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.linear_model import LinearRegression

df=pd.read_csv("https://raw.githubusercontent.com/Timjini/banking_marketing/refs/heads/main/data/bank.csv")

def clean_dataset_(data):
    """
    Cleans the given DataFrame by handling missing values,
    removing duplicates, and filtering out outliers.
    """
    # handeling missing values in the dataset
    data.fillna(data.mean(), inplace=True)

    # removing any duplicates
    data.drop_duplicates(inplace=True)

    # using zscore to handle outliers
    numeric_df = data.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(numeric_df))

    # use 3 as zscore threshold
    threshold = 3
    outlier_mask = (z_scores > threshold).any(axis=1)

    # remove rows with outliers
    clean_df = data[~outlier_mask]

    # return cleaned dataset
    return clean_df

cleaned_df=clean_dataset_(df)
print(cleaned_df.head(100))