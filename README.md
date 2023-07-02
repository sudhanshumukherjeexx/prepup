# 💻 Prepup
Built with ♥️ by [Sudhanshu Mukherjee](https://www.linkedin.com/in/sudhanshumukherjeexx/)

[![image](https://img.shields.io/pypi/v/prepup.svg)](https://pypi.python.org/pypi/prepup)
<!-- [![image](https://img.shields.io/conda/vn/conda-forge/prepup.svg)](https://anaconda.org/conda-forge/prepup) -->

### Prepup is a free open-source package that lets you inspect, explore, visualize, and perform pre-processing tasks on datasets in your windows/macOS terminal.

##  Installation
- Prepup can be installed using the Pip package manager.
- ### !pip install Prepup

## Motivation
- Developing an efficient and user-friendly command line tool for data pre-processing to handle various tasks such as missing data, data formatting, and cleaning, with a simple interface and scalability for large datasets.

## File Format Supported
-   CSV
-   EXCEL
-   PARQUET

## Why you should use Prepup?

### It's Superfast
-   Prepup is built on Polars which is an alternative to pandas and helps you load and manipulate the DataFrames faster.

### Analytical 
-   Prepup handles tasks ranging from the shape of data to the Standardizing of the feature before training the model. It does it right on the terminal.

### Compatible 
-   Prepup supports CSV, EXCEL, and PARQUET formats making it compatible to go with different file formats.

### Non-Destructive 
-   Prepup doesn't alter your raw data, It saves pre-processed data only when the user specifies the path.

### Lives in your Terminal
-   Prepup is terminal-based and has specific entry points designed for using it instantly.

# Command Line Arguments available in PREPUP

## 🕵️ Prepup "File name or File path" -inspect
File Name: If the current working directory is same as the file location or FILE PATH
https://github.com/sudhanshumukherjeexx/prepup/assets/64360018/93da36fc-1c7e-449c-9732-bfce81f3a915
-   inspect flag takes the dataframe and returns the Features available, Features datatype, and missing values present in the Dataset.

## 🧭 Prepup "File name or File path" -explore
File Name: If the current working directory is same as the file location or FILE PATH
https://github.com/sudhanshumukherjeexx/prepup/assets/64360018/eeccaf19-6c2a-4e8c-ab4a-8c3afb59f8c5
-   explore flag takes the dataframe and returns the Features available, Features datatype, Correlation between features, Detects Outliers, Checks Normal Distribution, Checks Skewness, Checks Kurtosis, and also allows the option to check if the dataset is Imbalanced.

## 📊 Prepup "File name or File path" -visualize
File Name: If the current working directory is same as the file location or FILE PATH
https://github.com/sudhanshumukherjeexx/prepup/assets/64360018/61fffd53-0b26-4537-ac1d-5296a2f8b52e
-   visualize flag plots of the feature distribution directly on the terminal.

## 🔥 Prepup "File name or File path" -impute
File Name: If the current working directory is same as the file location or FILE PATH
https://github.com/sudhanshumukherjeexx/prepup/assets/64360018/3d0160af-0059-4b4e-b278-abe8a587c5b5
- There are 8 different strategies available to impute missing data using Prepup

    - Option 1 - Drops the Missing Data
    - Option 2 - Impute Missing values with a Specific value
    - Option 3 - Impute Missing values with Mean.
    - Option 4 - Impute Missing values with Median.
    - Option 5 - Impute Missing value based on the distribution of existing columns.
    - Option 6 - Impute Missing values based on Forward Fill Strategy where missing values are imputed based on the previous data points.
    - Option 7 - Impute Missing values based on Backward Strategy where missing values are imputed based on the next data points.
    - Option 8 - Impute missing values based on K-Nearest Neighbors.

## 🌐 Prepup "File name or File path" -standardize
File Name: If the current working directory is same as the file location or FILE PATH
https://github.com/sudhanshumukherjeexx/prepup/assets/64360018/c098a7aa-1cb9-464b-bd89-1ea3c38b842e
-   Standardize allows you to standardize the dataset using two different methods:
    1. Robust Scaler

    2. Standard Scaler 

-   Robust Scaler is recommended if there are outliers present and you feel they can have an influence on the Machine Learning model.

-   Standard Scaler is the go-to function if you want to standardize the dataset before training the model on it.

# License
-   Free software: MIT license

# Package Link
-   Github: https://github.com/sudhanshumukherjeexx/prepup

-   Documentation: https://sudhanshumukherjeexx.github.io/prepup
