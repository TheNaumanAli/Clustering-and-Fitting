import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

def clean_and_convert(df, col):
    """Convert the specified column to numeric, coercing errors to NaN."""
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_relational_plot(df):
    """Create a relational plot showing relationship between two variables."""
    fig, ax = plt.subplots()
    x_col = "Age"
    y_col = "Ticket_Price"
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    sns.lineplot(data=df.sort_values(by=x_col), x=x_col, y=y_col, ax=ax, color='orange')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Relational Plot - Scatter with Line")
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close(fig)

def plot_categorical_plot(df):
    """Create a bar chart for categorical data."""
    fig, ax = plt.subplots()
    category_col = "Movie_Genre"
    df[category_col].value_counts().plot(kind='bar', ax=ax)
    plt.xlabel(category_col)
    plt.ylabel("Count")
    plt.title("Categorical Plot - Bar Chart")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close(fig)

def plot_statistical_plot(df):
    """Create a heatmap showing correlations between numerical variables."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Statistical Plot - Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    plt.close(fig)

def plot_elbow_method(df, col1, col2):
    """Plot the elbow method to find optimal k for KMeans."""
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        distortions.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')
    plt.savefig('elbow_plot.png')
    plt.close(fig)

def statistical_analysis(df, col: str):
    """Perform statistical analysis on a numerical column."""
    df = clean_and_convert(df, col).dropna(subset=[col])
    if df[col].empty:
        print(f"Warning: Column {col} is empty after conversion. Skipping analysis.")
        return None
    mean = df[col].mean()
    median = df[col].median()
    std_dev = df[col].std()
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    correlation = df.select_dtypes(include=[np.number]).corr()
    print(f"Descriptive statistics for {col}:")
    print(df[col].describe())
    print(df.head())
    print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "correlation": correlation
    }

def preprocessing(df):
    """Preprocess the dataframe by cleaning and analyzing."""
    df = df.dropna()
    numeric_df = df.select_dtypes(include=[np.number])
    print("Full Data Description:")
    print(df.describe())
    print("First few rows:")
    print(df.head())
    print("Numeric Correlation Matrix:")
    print(numeric_df.corr())
    return df

def perform_clustering(df, col1, col2):
    """Perform K-means clustering on two specified columns."""
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: Columns {col1} and/or {col2} not found in DataFrame.")
        return None, None, None, None, None
    data = df[[col1, col2]].dropna()
    if data.empty:
        print("Error: No valid data for clustering after dropping NaNs.")
        return None, None, None, None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    best_k = 3
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    print(f"Silhouette Score for k={best_k}: {score:.4f}")
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans, ykmeans = centroids_original[:, 0], centroids_original[:, 1]
    return labels, data, xkmeans, ykmeans, range(best_k)

def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot the results of clustering."""
    if labels is None:
        return
    fig, ax = plt.subplots()
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=100, label='Centroids')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.title("Clustering Plot")
    plt.savefig('clustering.png')
    plt.close(fig)

def perform_fitting(df, col1, col2):
    """Perform linear regression on two specified columns with scaling."""
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: Columns {col1} and/or {col2} not found in DataFrame.")
        return None, None, None
    data = df[[col1, col2]].dropna()
    if data.empty:
        print("Error: No valid data for fitting after dropping NaNs.")
        return None, None, None
    X = data[col1].values.reshape(-1, 1)
    y = data[col2].values.reshape(-1, 1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return data, X, y_pred

def plot_fitted_data(data, x, y):
    """Plot the results of linear regression."""
    if data is None:
        return
    fig, ax = plt.subplots()
    plt.scatter(x, data.iloc[:, 1], label='Actual Data', alpha=0.5)
    plt.plot(x, y, color='red', label='Fitted Line')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title("Regression Fit")
    plt.legend()
    plt.savefig('fitting.png')
    plt.close(fig)

def main():
    """Main function to execute the analysis pipeline."""
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Error: Not enough numeric columns for analysis.")
        return
    col1, col2 = numeric_cols[:2]
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col1)
    if moments:
        print(f"For the attribute {col1}: Mean = {moments['mean']:.2f}, Std Dev = {moments['std_dev']:.2f}, Skewness = {moments['skewness']:.2f}, Kurtosis = {moments['kurtosis']:.2f}")
    plot_elbow_method(df, col1, col2)
    clustering_results = perform_clustering(df, col1, col2)
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, col1, col2)
    plot_fitted_data(*fitting_results)

if __name__ == '__main__':
    main()