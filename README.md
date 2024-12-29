This repository contains a Python script for analyzing and modeling data related to tablet press performance and failure prediction. The project is developed by Sara Fakhri Shahidani and includes exploratory data analysis, visualization, and predictive modeling.
1. Data Overview
Reads and explores Tablet_press_data.csv, including:
Displaying data structure and summary statistics.
Checking for missing values, duplicates, and data types.
2. Data Visualization
Provides insightful visualizations to understand the distribution and relationship of features:
Box plots for feature distribution,
Histograms for data frequence,
Heatmap to display correlations among features.
3. Predictive Modeling
Implements Linear Regression to predict failure outcomes.
Splits data into training and testing sets.
Outputs performance metrics:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
4. Cross-Validation
Uses K-Fold Cross-Validation to evaluate model performance and compares results before and after applying K-Fold.
5. Comparison of Metrics
Visualizes the comparison of metrics (R², MSE, RMSE, MAE) before and after K-Fold using bar charts.
##Installation
Clone the repository:
git clone https://github.com/yourusername/tablet-press-analysis.git
###Install required Python libraries:
pip install pandas matplotlib seaborn scikit-learn numpy
###Place the Tablet_press_data.csv file in the same directory as the script.
Run the script using:python tablet.py
The script will:

-Load and analyze the data.
-Generate visualizations and display plots.
-Train a Linear Regression model and output performance metrics.
-Compare model performance with and without K-Fold Cross-Validation
###File Structure
-tablet.py: Main script for analysis and modeling.
-Tablet_press_data.csv: Dataset (needs to be provided).
###Outputs
Plots:
 *Boxplots and histograms for feature distributions.
 *Heatmap for correlation matrix.
 *Bar chart comparing metrics before and after K-Fold.
Metrics:
 *R², MAE, MSE, RMSE (for both before and after K-Fold).
###Contributing
If you would like to contribute to this project:

Fork the repository.
Create a new branch for your feature:
git checkout -b feature-name
Commit your changes:
git commit -m "Add your message"
Push to your branch:
git push origin feature-name
Create a pull request.
Author
This project is developed and maintained by Sara Fakhri Shahidani.





