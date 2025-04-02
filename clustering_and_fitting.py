# Import necessary libraries
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import seaborn as sns  # For enhanced visualizations
from sklearn.cluster import KMeans  # For clustering
from sklearn.preprocessing import StandardScaler  # For data scaling
from sklearn.linear_model import LinearRegression  # For linear regression
from sklearn.metrics import silhouette_score  # For clustering evaluation

def clean_and_convert(df, col):
    """Convert the specified column to numeric, coercing errors to NaN."""
    # Convert column to numeric, setting invalid values to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_relational_plot(df):
    """Create a relational plot showing relationship between two variables."""
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Define columns to plot (could be parameterized)
    x_col = "Age"  # or another meaningful numerical column
    y_col = "Ticket_Price"
    
    # Create scatter plot
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    # Add trend line
    sns.lineplot(data=df.sort_values(by=x_col), x=x_col, y=y_col, ax=ax, color='orange')
    
    # Set labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Relational Plot - Scatter with Line")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close(fig)  # Close figure to free memory

def plot_categorical_plot(df):
    """Create a bar chart for categorical data."""
    fig, ax = plt.subplots()
    category_col = "Movie_Genre"  # or any other meaningful categorical column
    
    # Create bar plot of value counts
    df[category_col].value_counts().plot(kind='bar', ax=ax)
    
    # Set labels and title
    plt.xlabel(category_col)
    plt.ylabel("Count")
    plt.title("Categorical Plot - Bar Chart")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close(fig)

def plot_statistical_plot(df):
    """Create a heatmap showing correlations between numerical variables."""
    fig, ax = plt.subplots(figsize=(10, 8))  # Larger figure size
    
    # Calculate correlation matrix for numerical columns
    corr = df.select_dtypes(include=[np.number]).corr()
    
    # Create heatmap with annotations
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Statistical Plot - Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    plt.close(fig)

def statistical_analysis(df, col: str):
    """Perform statistical analysis on a numerical column."""
    # Clean and convert the column to numeric
    df = clean_and_convert(df, col).dropna(subset=[col])
    
    # Check if column is empty after conversion
    if df[col].empty:
        print(f"Warning: Column {col} is empty after conversion. Skipping analysis.")
        return None

    # Calculate various statistics
    mean = df[col].mean()
    median = df[col].median()
    std_dev = df[col].std()
    skewness = df[col].skew()  # Measure of asymmetry
    kurtosis = df[col].kurtosis()  # Measure of tail heaviness
    correlation = df.select_dtypes(include=[np.number]).corr()  # Correlation matrix

    # Print descriptive statistics
    print(f"Descriptive statistics for {col}:")
    print(df[col].describe())
    print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

    # Return statistics as a dictionary
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
    # Drop rows with missing values
    df = df.dropna()
    
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Print descriptive statistics
    print("Full Data Description:")
    print(df.describe())
    
    # Print correlation matrix
    print("Numeric Correlation Matrix:")
    print(numeric_df.corr())
    
    return df

def perform_clustering(df, col1, col2):
    """Perform K-means clustering on two specified columns."""
    # Check if columns exist in dataframe
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: Columns {col1} and/or {col2} not found in DataFrame.")
        return None, None, None, None, None

    # Select relevant data and drop missing values
    data = df[[col1, col2]].dropna()
    
    # Check if data is empty after cleaning
    if data.empty:
        print("Error: No valid data for clustering after dropping NaNs.")
        return None, None, None, None, None

    # Scale the data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Set number of clusters (could be determined dynamically)
    best_k = 3
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled_data)

    # Convert centroids back to original scale
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans, ykmeans = centroids_original[:, 0], centroids_original[:, 1]

    return labels, data, xkmeans, ykmeans, range(best_k)

def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot the results of clustering."""
    if labels is None:
        return

    # Create figure
    fig, ax = plt.subplots()
    
    # Plot data points colored by cluster
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    
    # Plot cluster centroids
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=100, label='Centroids')
    
    # Set labels and legend
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.title("Clustering Plot")
    
    # Save plot
    plt.savefig('clustering.png')
    plt.close(fig)

def perform_fitting(df, col1, col2):
    """Perform linear regression on two specified columns."""
    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: Columns {col1} and/or {col2} not found in DataFrame.")
        return None, None, None

    # Select and clean data
    data = df[[col1, col2]].dropna()
    if data.empty:
        print("Error: No valid data for fitting after dropping NaNs.")
        return None, None, None

    # Prepare data for regression
    X = data[col1].values.reshape(-1, 1)  # Feature matrix
    y = data[col2].values  # Target vector
    
    # Create and fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    y_pred = model.predict(X)

    return data, X, y_pred

def plot_fitted_data(data, x, y):
    """Plot the results of linear regression."""
    if data is None:
        return

    # Create figure
    fig, ax = plt.subplots()
    
    # Plot actual data points
    plt.scatter(x, data.iloc[:, 1], label='Actual Data', alpha=0.5)
    
    # Plot regression line
    plt.plot(x, y, color='red', label='Fitted Line')
    
    # Set labels and legend
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title("Regression Fit")
    plt.legend()
    
    # Save plot
    plt.savefig('fitting.png')
    plt.close(fig)

def main():
    """Main function to execute the analysis pipeline."""
    # Load data
    df = pd.read_csv('data.csv')
    
    # Preprocess data
    df = preprocessing(df)
    
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Check if we have enough numerical columns
    if len(numeric_cols) < 2:
        print("Error: Not enough numeric columns for analysis.")
        return

    # Select first two numerical columns for analysis
    col1, col2 = numeric_cols[:2]
    
    # Generate various plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    # Perform statistical analysis
    moments = statistical_analysis(df, col1)
    if moments:
        print(f"For the attribute {col1}: Mean = {moments['mean']:.2f}, Std Dev = {moments['std_dev']:.2f}, Skewness = {moments['skewness']:.2f}, Kurtosis = {moments['kurtosis']:.2f}")

    # Perform clustering and plot results
    clustering_results = perform_clustering(df, col1, col2)
    plot_clustered_data(*clustering_results)
    
    # Perform regression and plot results
    fitting_results = perform_fitting(df, col1, col2)
    plot_fitted_data(*fitting_results)

if __name__ == '__main__':
    # Execute main function when script is run directly
    main()