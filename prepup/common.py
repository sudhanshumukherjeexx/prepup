"""This module contains functions and classes used by the other modules.

    author: "Neokai"
"""
import polars as pl
import pandas as pd
import plotext as tpl
import numpy as np
import io
import nbformat as nbf
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from termcolor import colored
from pyfiglet import Figlet



term_font = Figlet(font="term")

class Prepup:
    
    def __init__(self, dataframe):
        """
        The __init__ function is called when the class is instantiated.
        It takes a dataframe as an argument and assigns it to self.dataframe, which makes it available to other functions in the class.
        
        :param self: Represent the instance of the class
        :param dataframe: Pass the dataframe to the class
        :return: An instance of the class
        :author: Neokai
        """
        self.dataframe = dataframe
    
    def features_available(self):
        """
        The features_available function returns a list of the features available in the dataframe.
                
        
        :param self: Represent the instance of the class
        :return: A list of the column names in the dataframe
        :author: Neokai
        """
        #try:
        return self.dataframe.columns
        #except:
            #print("Something went wrong...\nCouldn't intialize the dataset properly...")

    def dtype_features(self):
        """
        The dtype_features function returns the data types of each feature in the dataset.
            This is useful for determining which features are categorical and which are numerical.
        
        :param self: Represent the instance of the class
        :return: A series with the data type of each feature (column) in a pandas dataframe
        :author: Neokai
        """
        # try:
        return self.dataframe.dtypes
        # except:
        #     print("Something went wrong....\nCouldn't display DataTypes of Features...")

    def missing_values(self):
        """
        The missing_values function returns the number of missing values in each column.
            Args:
                self (DataFrame): The DataFrame object to be analyzed.
            Returns:
                A dictionary with the columns as keys and their respective null counts as values.
        
        :param self: Represent the instance of the class
        :return: The number of missing values in each column
        :author: Neokai
        """
        #try:
        if self.dataframe.is_empty() == True:
            print("No Missing Value Found")
        else:
            missing_value = self.dataframe.null_count()
            return missing_value
        # except:
        #     print("Something went wrong....\nChecking Dataset: Recommended.\nPlease try again")
    
    def shape_data(self):
        """
        The shape_data function returns the shape of the dataframe.
                :return: The shape of the dataframe as a tuple (rows, columns)
        
        :param self: Represent the instance of the class
        :return: The shape of the dataframe
        :author: Neokai
        """
        #try:
        return self.dataframe.shape
        # except:
        #     print("Something went wrong....\nChecking Dataset: Recommended.\nPlease try again")
    
    def missing_plot(self):
        """
        The missing_plot function takes in a dataframe and plots the missing values for each column.
            It also prints out the number of missing values for each column.
        
        :param self: Represent the instance of the class
        :return: The number of missing values for each column in the dataframe
        :author: Neokai
        """
        #try:
        empty_count = 0
        non_empty_count = 0
        for column in self.dataframe.columns:
            val = self.dataframe.select(pl.col([column]).is_null().any())
            if val[column][0] == True:
                empty_count += 1
            else:
                non_empty_count += 1
        if empty_count == 0:
            print(colored(term_font.renderText("No Missing Value Found"), 'green+'))
            #print("No Missing Value Found")
        else:
            missing_counts = self.dataframe.null_count()
            missing_counts = missing_counts.to_dicts()
            df = pd.DataFrame(missing_counts)
            melt_df = pd.melt(df, value_vars=df.columns)
            melt_df.columns = ['Features', 'Missing_Value_Count']
            df_new = melt_df.loc[melt_df['Missing_Value_Count'] != 0]
            print(df_new,"\n")
            tpl.simple_bar(df_new['Features'], df_new['Missing_Value_Count'], width=100, title='Missing Value Count in Each Feature', color='red+')
            tpl.theme('matrix')
            tpl.show()
            tpl.clear_data()
        # except:
        #     print("Something went wrong....\nFailed to fetch missing value count.\nPlease try again...")
            
    def plot_histogram(self):
        """
        The plot_histogram function plots a histogram for each numerical column in the dataframe.
            The function takes no arguments and returns nothing.
        
        :param self: Represent the instance of the class
        :return: A histogram for each of the columns in the dataframe
        :author: Neokai
        """
        #try:
        dataframe = self.dataframe.fill_null(0)
        for column in self.dataframe.columns:
            if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                tpl.clear_data()
                tpl.theme('dark')
                tpl.plotsize(80,20)
                print("\n")
                tpl.hist(dataframe[column], bins=20,color='light-blue', marker='dot') #color=46)
                tpl.title(column)
                #tpl.grid(horizontal=50, vertical=50)
                tpl.show()
                tpl.clear_data()
        # except:
        #     print("Something went wrong....\nFailed to fetch feature distribution...\nPlease try again...")

    

    
    def correlation_n(self):
        """
        The correlation_n function takes in a dataframe and returns the correlation between all numerical features.
            The function first selects only the numerical columns from the dataframe, then it creates two lists: one for 
            feature pairs and another for their corresponding correlation values. It then uses simple_bar to plot these 
            values as a bar graph.
        
        :param self: Represent the instance of the class
        :return: A bar graph of the correlation between all numerical features in the dataset
        :author: Neokai
        """
        #try:
        dtype_select_df = self.dataframe.select([pl.col(pl.Decimal),pl.col(pl.Float32),pl.col(pl.Float64),pl.col(pl.Int16),pl.col(pl.Int32),pl.col(pl.Int64),pl.col(pl.Int8),pl.col(pl.UInt16),pl.col(pl.UInt32),pl.col(pl.UInt64),pl.col(pl.UInt8),pl.col(pl.Date),pl.col(pl.Datetime),pl.col(pl.Duration),pl.col(pl.Time)])
        dtype_select_df = dtype_select_df.to_pandas()
        #corr_d = pd.DataFrame()
        features = []
        correlation_val = []
        for i in dtype_select_df.columns:
            for j in dtype_select_df.columns:
                feature_pair = i,j
                features.append(feature_pair)
                correlation_val.append(round(dtype_select_df[i].corr(dtype_select_df[j]),2))
        tpl.simple_bar(features, correlation_val,width=100, title='Correlation Between these Features', color=92,marker='*')
        tpl.show()
        tpl.clear_data()
        # except:
        #     print("Something went wrong....\nFailed to fetch Correlation Matrix...\nPlease try again...")

    def scatter_plot(self):
        """
        The scatter_plot function takes the dataframe and selects all columns that are numeric.
        It then creates a scatter plot for each pair of numeric columns in the dataframe.
        
        :param self: Represent the instance of the class
        :return: A scatter plot for each column in the dataframe
        :author: Neokai
        """
        #try:
        dtype_select_df = self.dataframe.select([pl.col(pl.Decimal),pl.col(pl.Float32),pl.col(pl.Float64),pl.col(pl.Int16),pl.col(pl.Int32),pl.col(pl.Int64),pl.col(pl.Int8),pl.col(pl.UInt16),pl.col(pl.UInt32),pl.col(pl.UInt64),pl.col(pl.UInt8),pl.col(pl.Date),pl.col(pl.Datetime),pl.col(pl.Duration),pl.col(pl.Time)])
        dtype_select_df = dtype_select_df.to_pandas()
        for i in dtype_select_df.columns:
            for j in dtype_select_df.columns:
                scatter_p = pd.DataFrame()
                scatter_p["value1"] = dtype_select_df[[i]] 
                scatter_p["value2"] = dtype_select_df[[j]]
                tpl.theme('matrix')
                tpl.plotsize(80,20)
                tpl.title("\nDistribution of {0} vs {1}".format(i,j))
                tpl.scatter(scatter_p["value1"], scatter_p["value2"], color='white')
                tpl.show()
                tpl.clear_data()
        # except:
        #     print("Something went wrong....\nScatter Plot Build Failed...\nPlease try again...")
    
    def find_outliers(self, k=1.5):
        """
        The find_outliers function takes a dataframe and returns the outliers in each column.
            The function uses the interquartile range to determine if a value is an outlier or not.
            The default k value is 1.5, but can be changed by passing in another float as an argument.
        
        :param self: Represent the instance of the class
        :param k: Calculate the iqr (interquartile range)
        :return: A print statement of the outliers detected in each column
        :author: Neokai
        """
        #try:
        dtype_select_df = self.dataframe.select([pl.col(pl.Decimal),pl.col(pl.Float32),pl.col(pl.Float64),pl.col(pl.Int16),pl.col(pl.Int32),pl.col(pl.Int64),pl.col(pl.Int8),pl.col(pl.UInt16),pl.col(pl.UInt32),pl.col(pl.UInt64),pl.col(pl.UInt8),pl.col(pl.Date),pl.col(pl.Datetime),pl.col(pl.Duration),pl.col(pl.Time)])
        dtype_select_df = dtype_select_df.to_pandas()

        for i in dtype_select_df.columns:
            outliers = []
            q1 = np.percentile(dtype_select_df[i].values, 25)
            q3 = np.percentile(dtype_select_df[i].values, 75)
            iqr = q3 - q1
            lower_bound = q1 - k*iqr
            upper_bound = q3 + k*iqr
            for j in dtype_select_df[i].values:
                if j < lower_bound or j > upper_bound:
                    outliers.append(j)
            print(f"\tOutliers detected in {i}\n")
        # except:
        #     print("Something went wrong....\nFailed to Find Outliers...\nPlease try again...")
    
    
    def feature_scaling(self):
        """
        The feature_scaling function normalizes the dataframe by subtracting the mean and dividing by standard deviation.
            The function takes in a dataframe as input, drops the target variable if it is specified, 
            converts to pandas dataframe and then performs feature scaling on all columns that are not categorical or boolean. 
            It then saves this normalized dataset as a csv file at user-specified path.
        
        :param self: Represent the instance of the class
        :return: A dataframe with normalized values
        :author: Neokai
        """
        #try:
        isExist = os.path.exists("missing_data.parquet")
        if isExist == True:
            dataframe = pl.read_parquet("missing_data.parquet")
        else:
            dataframe = self.dataframe
        target_col = input("\nEnter the Target variable to drop: (or 'None')")
        
        if target_col != "None":
            df = dataframe.to_pandas()
            df = df.drop(target_col, axis=1)
        else:
            df = dataframe.to_pandas()
        
        for column in df.columns:
            if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                scaler = StandardScaler()
                df[[column]] = scaler.fit_transform(df[[column]])
        
        data_path = input("\nEnter path to save normalized data : ")
        path = data_path  
        s = "\\"
        if s in path:
            path = path.replace(os.sep, '/')
            path = path + "/NormalizedData.csv" 
            path = str(path)
            print(path)
        else:
            path = path + "/NormalizedData.csv"
        df.to_csv(path)
        print("\nFeature Normalized and saved succesfully")
        # except:
        #     print("Something went wrong....\nFailed to perform Feature Scaling...\nPlease try again...\nImputing missing values recommended.")
    
    def check_nomral_distrubution(self):
        """
        The check_nomral_distrubution function checks if the dataframe is normally distributed.
            It does this by using the Shapiro-Wilk test to check for normality. 
            The function will print out a message stating whether or not each column in the dataframe is normally distributed.
        
        :param self: Represent the instance of the class
        :return: The name of the column, whether it is normally distributed or not and its p-value
        :author: Neokai
        """
        #try:
        dataframe = self.dataframe
        for column in dataframe.columns:
            if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                stats, p_value = shapiro(dataframe[column])
                if p_value > 0.05:
                    h_8 = Figlet(font='term') 
                    print(colored(h_8.renderText(f"* {column} is Normally Distributed with a p-value of {p_value:.2f}\n"),'green'))
                else:
                    h_8 = Figlet(font='term')
                    print(colored(h_8.renderText(f"* {column} doesn't have a Normal Distribution with a p-value of {p_value:.8f} \n"), 'red'))
        # except:
        #     print("Something went wrong....\nFailed to perform Feature Scaling...\nPlease try again...")


    def imbalanced_dataset(self):    
        """
        The imbalanced_dataset function takes in a dataframe and the target variable as input.
        It then plots a bar graph of the distribution of the target variable.
        
        :param self: Represent the instance of the class
        :return: A bar plot of the target variable distribution
        :author: Neokai
        """
        dataframe = self.dataframe.to_pandas()
        val = input("Enter the Target Variable: ")
        
        target_dist = dataframe[val].value_counts()
        tpl.simple_bar(target_dist.index, target_dist.values, width=100,title='Target Variable Distribution',color=92)
        tpl.show()
        tpl.clear_data()
    

    def handle_missing_values(self):
        """
        The handle_missing_values function is used to handle missing values in the dataset.
        The user can choose from a variety of options to impute missing data, or drop it altogether.
        
        
        :param self: Represent the instance of the class
        :return: The dataframe with missing values imputed
        :author: Neokai
        """
        #try:
        dataframe = self.dataframe
        print("Choice Available to Impute Missing Data: \n")
        print("\t1. [Press 1] Drop Missing Data.\n")
        print("\t2. [Press 2] Impute Missing Data with Specific Value.\n")
        print("\t3. [Press 3] Impute Missing Data with Mean.\n")
        print("\t4. [Press 4] Impute Missing Data with Median.\n")
        print("\t5. [Press 5] Impute Missing Data based on Distribution of each Feature..\n")
        print("\t6. [Press 6] Impute Missing Data with Fill Forward Strategy.\n")
        print("\t7. [Press 7] Impute Missing Data with Backward Fill Strategy.\n")
        print("\t8. [Press 8] Impute Missing Data with Nearest Neighbours (Advisable if dataset has missing values randomly).\n")
        choice = int(input("\nEnter your choice: "))
        if choice == 1:
            dataframe = dataframe.drop_nulls()
            dataframe.write_parquet("missing_data.parquet")
        elif choice == 2:
            mv = int(input("Enter the value to replace missing data: "))
            dataframe = dataframe.fill_null(mv)
            dataframe.write_parquet("missing_data.parquet")
        elif choice == 3:
            dataframe = dataframe.to_pandas()
            for column in dataframe.columns:
                if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                    dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
            dataframe.to_parquet("missing_data.parquet")
        elif choice == 4:
            dataframe = dataframe.to_pandas()
            for column in dataframe.columns:
                if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                    dataframe[column] = dataframe[column].fillna(dataframe[column].median())
            dataframe.to_parquet("missing_data.parquet")
        elif choice == 5:
            dataframe = dataframe.to_pandas()
            for column in dataframe.columns:
                if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                    mean = dataframe[column].mean()
                    std = dataframe[column].std()
                    random_values = np.random.normal(loc=mean, scale=std, size=dataframe[column].isnull().sum())
                    dataframe[column] = dataframe[column].fillna(pd.Series(random_values,index=dataframe[column][dataframe[column].isnull()].index))
            dataframe.to_parquet("missing_data.parquet")
        elif choice == 6:
            dataframe = dataframe.fill_null(strategy="forward")
            dataframe.write_parquet("missing_data.parquet")
        elif choice == 7: 
            dataframe = dataframe.fill_null(strategy="backward")
            dataframe.write_parquet("missing_data.parquet")
        elif choice == 8: 
            dataframe = dataframe.to_pandas()
            for column in dataframe.columns:
                if dataframe[column].dtype != pl.Categorical and dataframe[column].dtype != pl.Utf8 and dataframe[column].dtype != pl.Boolean and dataframe[column].dtype != pl.Null and dataframe[column].dtype != pl.Object and dataframe[column].dtype != pl.Unknown:
                    missing_inds = dataframe[column].isnull()
                    non_missing_inds = ~missing_inds
                    non_missing_vals = dataframe[column][non_missing_inds]
                    closest_inds = np.abs(dataframe[column][missing_inds].values - non_missing_vals.values.reshape(-1,1)).argmin(axis=0)
                    dataframe.loc[missing_inds, column] = non_missing_vals.iloc[closest_inds].values
            dataframe.to_parquet("missing_data.parquet")
        print(colored(term_font.renderText("\nDone...")))
        # except:
        #     print("Something went wrong....\nFailed to perform Feature Scaling...\nPlease try again...")

        
        

class NotebookCreator:
    def __init__(self):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class with a file_path attribute, which will be used to store
        the path to a file that we want to read from or write to.
        
        :param self: Represent the instance of the class
        :return: Nothing
        :doc-author: Neokai
        """
        self.file_path = ""

    def load_file(self):
        """
        The load_file function takes the file path and returns a string that can be used to load the data into a pandas DataFrame.
            The function checks for the file extension and returns an appropriate string.
        
        :param self: Represent the instance of the class
        :return: A string
        :doc-author: Neokai
        """
        # Check file extension
        if self.file_path.endswith('.csv'):
            return f'pd.read_csv(\'{self.file_path}\')'
        elif self.file_path.endswith('.xlsx'):
            return f'pd.read_excel(\'{self.file_path}\')'
        elif self.file_path.endswith('.parquet'):
            return f'pd.read_parquet(\'{self.file_path}\')'
        else:
            raise ValueError("Invalid file format. Only CSV, Excel and Parquet files are supported.")

    def create_notebook(self):
        """
        The create_notebook function takes a file path as input and creates an ipython notebook with the following features:
            1. Install Dependencies
            2. Import Libraries
            3. Read the dataset 
            4. View the dataset 
            5. Features present in the dataset
            6. Datatypes of each feature in the dataset
                - Numeric, Categorical, Boolean, DateTime etc.,
            7. Impute Missing Values
            8. Perform Feature Scaling  
        
        :param self: Represent the instance of the class
        :return: A notebook object
        :doc-author: Neokai
        """
        
        self.file_path = input("Enter the File Path : ")
        
        df_n = self.load_file()
        s = "\\"
        if s in df_n:
            df = df_n.replace(os.sep, '/')
        else:
            df = df_n 

        # Create a new notebook object
        nb = nbf.v4.new_notebook()

        #Install Dependencies
        install_dependencies = ['pip install pandas numpy plotly matplotlib seaborn scipy sklearn']
        ins_d = nbf.v4.new_code_cell(''.join(install_dependencies))
        nb['cells'].append(ins_d)
        
        # Import pandas, numpy, and Plotly
        imports = ['#Import Libraries\nimport pandas as pd\n', 'import numpy as np\n', 'import plotly.express as px\n','import matplotlib.pyplot as plt\n','import scipy.stats as stats\n', 'import seaborn as sns\n', 'from sklearn.preprocessing import StandardScaler']
        import_cell = nbf.v4.new_code_cell(''.join(imports))
        nb['cells'].append(import_cell)

        #Read the dataset
        read_cell = nbf.v4.new_code_cell(f'df={df}')
        nb['cells'].append(read_cell)

        #view the dataset
        show_cell = nbf.v4.new_code_cell('df.head()')
        nb['cells'].append(show_cell)

        #Features present in the dataset
        feature_cell = nbf.v4.new_code_cell('df.columns')
        nb['cells'].append(feature_cell) 

        #Datatypes of the dataset
        info_cell = nbf.v4.new_code_cell('df.info()')
        nb['cells'].append(info_cell)

        #Shape of Data
        shape_cell = nbf.v4.new_code_cell('df.shape')
        nb['cells'].append(shape_cell)

        #Describe the dataset
        describe_cell = nbf.v4.new_code_cell('df.describe()')
        nb['cells'].append(describe_cell)

        #Feature Correlation
        corr_cell = nbf.v4.new_code_cell('df.corr()')
        nb['cells'].append(corr_cell)

        #Features: Missing value and their count
        feature_cell = nbf.v4.new_code_cell('df.isnull().sum()')
        nb['cells'].append(feature_cell)

        #Correlation Heatmap
        corr_hm = ['#Feature Correlation Heatmap\nfig = px.imshow(df.corr(), text_auto=True, color_continuous_scale="Blues", title="Feature Correlation Heatmap", aspect="auto")\nfig.show()']
        corr_cell_hm = nbf.v4.new_code_cell(corr_hm) 
        nb['cells'].append(corr_cell_hm)

        #Identify Outliers
        # Select only the numeric columns
        num_col =["#Identify Outliers using BoxPlot\nnumeric_cols = df.select_dtypes(include=['float64', 'int64']).columns\nfor col in numeric_cols:\n\tfig=px.box(df, y=col,title=f'Box Plot of {col} column')\n\tfig.show()"]
        num_col_io = nbf.v4.new_code_cell(num_col)
        nb['cells'].append(num_col_io)

        num_col_hist_qq =["#Feature Distribution\nnumeric_cols = df.select_dtypes(include=['float64', 'int64']).columns\nfor col in numeric_cols:\n\tplt.figure(figsize=(14,4)),\n\tplt.subplot(121)\n\tsns.kdeplot(df[col],shade=True,color='red')\n\tplt.title(col)\n\n\tplt.subplot(122)\n\tstats.probplot(df[col],dist='norm',plot=plt)\n\tplt.title(col)\n\tplt.show();"]
        num_col_io_hist_qq = nbf.v4.new_code_cell(num_col_hist_qq)
        nb['cells'].append(num_col_io_hist_qq)

        #Imputing Missing Values
        missing_val = ["#Impute Missing Values with Different Methods\ndf_mean = df.fillna(df.mean(numeric_only=True))\ndf_median = df.fillna(df.median(numeric_only=True))\ndf_ffill=df.fillna(method='ffill')\ndf_bfill=df.fillna(method='bfill')"]
        missing_val_cell = nbf.v4.new_code_cell(missing_val)
        nb['cells'].append(missing_val_cell)

        #Feature Scaling
        feature_scaling =["#Feature Scaling\nscaler = StandardScaler()\nnumeric_cols = df.select_dtypes(include=['float64', 'int64']).columns\nfor col in numeric_cols:\n\tdf[[col]]=scaler.fit_transform(df[[col]])"]
        feature_scaling_col = nbf.v4.new_code_cell(feature_scaling)
        nb['cells'].append(feature_scaling_col)

        #view the dataset
        show_cell = nbf.v4.new_code_cell('df.head()')
        nb['cells'].append(show_cell)
        

        data_path = input("\nEnter path to save normalized data : ")
        path = data_path  
        s = "\\"
        if s in path:
            path = path.replace(os.sep, '/')
            path = path + "/DataPreprocessed.ipynb" 
            path = str(path)
            print(path)
        else:
            path = path + "/DataPreprocessed.ipynb"

        # Write notebook to file
        with io.open(path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)