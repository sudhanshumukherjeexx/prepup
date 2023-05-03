# prepup


[![image](https://img.shields.io/pypi/v/prepup.svg)](https://pypi.python.org/pypi/prepup)
[![image](https://img.shields.io/conda/vn/conda-forge/prepup.svg)](https://anaconda.org/conda-forge/prepup)


![image](C:\Users\Asus\prepup\logo.png)

### Prepup is a free open-source package that lets you inspect, explore, visualize, and perform pre-processing tasks on datasets in your computer’s terminal.

## Installation
-   Prepup can be installed using the Pip package manager.

### !pip install prepup

## Motivation
- Developing an efficient and user-friendly command line tool for data pre-processing to handle various tasks such as missing data, data formatting, and cleaning, with a simple interface and scalability for large datasets.

## File Format Supported
-   CSV
-   EXCEL
-   PARQUET

## Why you should use Prepup?

### It's Super fast
-   Prepup is built on Polars which is alternative to pandas and helps you load and manipulate the DataFrames faster.

### Analytical 
-   Prepup handles tasks ranging from shape of data to the Standardizing the feature before training the model. It does it right on the terminal.

### Compatible 
-   Prepup supports CSV, EXCEL and PARQUET formats making it compatible to go with different file formats.

### Non Destructive 
-   Prepup doesn't alters your raw data, It saves pre-processed data only when user specifies the path.

### Lives in your Terminal
-   Prepup is terminal based and has specific entry points designed for using it instantly.

# Command Line Arguments available in PREPUP

## prepup <File name or File path> -inspect

-   inspect flag takes the dataframe and returns the Features available, Features datatype and missing values present in the Dataset.

## prepup <File name or File path> -explore

-   explore flag takes the dataframe and returns the Features available, Features datatype, Correlation between features, Detects Outliers, Checks Normal Distribution, Checks Skewness, Checks Kurtosis and also allows the option to check if the dataset is Imbalanced.

## prepup <File name or File path> -explore

-   visualize flag plots the feature distribution directly on the terminal.

## prepup <File name or File path> -visualize

-   visualize flag plots the feature distribution directly on the terminal.

## prepup <File name or File path> -impute
- There are 8 different strategies available to impute missing data using Prepup

    - Option 1 – Drops the Missing Data
    - Option 2 – Impute Missing values with a Specific value
    - Option 3 – Impute Missing values with Mean.
    - Option 4 – Impute Missing values with Median.
    - Option 5 – Impute Missing value based on the distribution of existing columns.
    - Option 6 – Impute Missing values based on Forward Fill Strategy where missing values are imputed based on the previous data points.
    - Option 7 - Impute Missing values based on Backward Strategy where missing values are imputed based on the next data points.
    - Option 8 – Impute missing values based on K-Nearest Neighbors.

## prepup <File name or File path> -standardize

-   Standardize allows you to standardize the dataset using two different methods:
    1. Robust Scaler

    2. Standard Scaler 

-   Robust Scaler is recommended if there are outliers present and you feel they can have influence on the Machine Learning model.

-   Standard Scaler is go to function if you want to standardize the dataset before training the model on it.


# License

-   Free software: MIT license

# Package Link
-   Github: https://github.com/sudhanshumukherjeexx/prepup

-   Documentation: https://sudhanshumukherjeexx.github.io/prepup
    
