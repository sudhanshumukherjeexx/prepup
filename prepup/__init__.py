"""Main module.

   author : "Neokai"
"""
import argparse
import polars as pl
from prepup.common import Prepup
from prepup.common import NotebookCreator
from termcolor import colored
from pyfiglet import Figlet
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Prepup is a free open-source package that lets you inspect, explore, visualize, and perform pre-processing tasks on datasets in your computers terminal.")
    parser.add_argument('file',type=str, help='Dataset file')
    parser.add_argument('-inspect', action='store_true', help='Observe the dataset and its Features.')
    parser.add_argument('-explore', action='store_true', help='Explore Dataset.')
    parser.add_argument('-visualize', action='store_true', help='Visualize Feature Distribution.')
    parser.add_argument('-impute', action='store_true', help='Impute Missing values.')
    parser.add_argument('-standardize', action='store_true', help='Standardize Feature Columns.')
    parser.add_argument('-rii', action='store_true', help='Convert code to Python Notebook.')
    # parser.add_argument('-fmissing', action='store_true', help='Handle Missing data')
    # parser.add_argument('-fhist', action='store_true', help='Feature Distribution in the Dataset')
    # parser.add_argument('-fdist', action='store_true', help='Plots Features Distribution')
    # parser.add_argument('-fscaling', action='store_true', help='Feature Scaling')
    # parser.add_argument('-nd', action='store_true', help='Check Normal Distribution')
    # parser.add_argument('-id', action='store_true', help='Check if Dataset is Imbalanced.')
    # parser.add_argument('-pnotebook', action='store_true', help='Creates a Jupyter Notebook of all the functions displayed on Terminal')
    # parser.add_argument('-rii', action='store_true', help='Global Dataset')
    return parser.parse_args()

def load_file(file_path):
    # Check file extension
    if file_path.endswith('.csv'):
        return pl.read_csv(file_path, ignore_errors=True)
    elif file_path.endswith('.xlsx'):
        return pl.read_excel(file_path, ignore_errors=True)
    elif file_path.endswith('.parquet'):
        return pl.read_parquet(file_path)
    else:
        raise ValueError("Invalid file format. Only CSV and Excel files are supported.")


def main():
    args = parse_args()
    intro = Figlet(font='big')
    print(colored(intro.renderText("PREPUP !"), 'green'))
    time.sleep(0.5)
    df = load_file(args.file)
    
    if args.standardize:
        start = time.time() 
        crafter = Prepup(df)
        
        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nFeature Scaling in Progress..."),'light_blue'))
        time.sleep(0.2)
        crafter.feature_scaling()
        
        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -normalize flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.explore:
        start = time.time() 

        crafter = Prepup(df)
        h_1 = Figlet(font='term')
        print(colored(h_1.renderText("Features present in the Dataset..."), 'light_blue')) 
        time.sleep(0.5)
        print(crafter.features_available())
        
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nDatatype of each Feature..."), 'light_blue'))
        time.sleep(0.5)
        print(crafter.dtype_features())

        h_5 = Figlet(font='term')
        print(colored(h_5.renderText("\nFeature Correlation..."),'light_blue'))
        time.sleep(0.5)
        crafter.correlation_n()

        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nCheck for Normal Distribution..."),'light_blue'))
        time.sleep(0.2)
        crafter.check_nomral_distrubution()

        h_9 = Figlet(font='term')
        print(colored(h_9.renderText("\nDetecting Outliers..."),'light_blue'))
        crafter.find_outliers()

        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Skewness Present in the Dataset..."),'light_blue'))
        time.sleep(0.5)
        crafter.skewness()

        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Skewness Present in the Dataset..."),'light_blue'))
        time.sleep(0.5)
        crafter.kurtosis()

        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nCheck for Imabalanced Dataset..."),'light_blue'))
        time.sleep(0.2)
        crafter.imbalanced_dataset()

        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -explore flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.visualize:
        start = time.time() 
        crafter = Prepup(df)
        
        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nFeature Distribution..."),'light_blue'))
        time.sleep(0.2)
        crafter.plot_histogram()

        # h_6 = Figlet(font='term')
        # print(colored(h_6.renderText("\nRelationship visualized between two variables (Pair-Plot)..."),'light_blue'))
        # crafter =  Prepup(df)
        # crafter.scatter_plot()

        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -visualize flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.impute:
        start = time.time()
        crafter =  Prepup(df)

        h_6 = Figlet(font='term')
        print(colored(h_6.renderText("\nHandle Missing Data..."),'light_blue'))
        time.sleep(0.2)
        crafter.handle_missing_values()

        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for -impute flag..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.inspect:
        start = time.time() 
        crafter = Prepup(df)
        h_1 = Figlet(font='term')
        print(colored(h_1.renderText("Features Present in the Dataset..."), 'light_blue')) 
        time.sleep(0.5)
        print(crafter.features_available())
        
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nFeature's Datatype..."), 'light_blue'))
        time.sleep(0.5)
        print(crafter.dtype_features())

        h_3 = Figlet(font='term')
        print(colored(h_3.renderText("\nShape of Data..."),'light_blue'))
        time.sleep(0.5)
        print(crafter.shape_data())
                
        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Missing values count..."),'light_blue'))
        time.sleep(0.5)
        crafter.missing_plot()
        
        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for loading and inspecting data..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
    elif args.rii:
        notebook_creator=NotebookCreator()
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nCreating Jupyter Notebook..."), 'light_blue'))
        time.sleep(0.5)
        notebook_creator.create_notebook()
    else:
        start = time.time() 
        crafter = Prepup(df)
        h_1 = Figlet(font='term')
        print(colored(h_1.renderText("Features Present in the Dataset..."), 'light_blue')) 
        time.sleep(0.5)
        print(crafter.features_available())
        
        h_2 = Figlet(font='term')
        print(colored(h_2.renderText("\nFeature's Datatype..."), 'light_blue'))
        time.sleep(0.5)
        print(crafter.dtype_features())

        h_3 = Figlet(font='term')
        print(colored(h_3.renderText("\nShape of Data..."),'light_blue'))
        time.sleep(0.5)
        print(crafter.shape_data())
                
        h_4 = Figlet(font='term')
        print(colored(h_4.renderText("\nFeatures: Missing values count..."),'light_blue'))
        time.sleep(0.5)
        crafter.missing_plot()
        
        h_8 = Figlet(font='term')
        print(colored(h_8.renderText("\nExecution Time for loading and inspecting data..."),'light_blue'))
        end_n = time.time()-start
        print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))



    # if args.fdist:
    #     h_6 = Figlet(font='term')
    #     print(colored(h_6.renderText("\nRelationship visualized between two variables..."),'light_blue'))
    #     crafter =  Prepup(df)
    #     crafter.scatter_plot()
    # elif args.fhist:
    #     crafter =  Prepup(df)
    #     h_6 = Figlet(font='term')
    #     print(colored(h_6.renderText("\nDistribution of Each Feature..."),'light_blue'))
    #     time.sleep(0.5)
    #     crafter.plot_histogram()
    # elif args.pnotebook:
    #     notebook_creator=NotebookCreator()
    #     h_2 = Figlet(font='term')
    #     print(colored(h_2.renderText("\nCreating Jupyter Notebook..."), 'light_blue'))
    #     time.sleep(0.5)
    #     notebook_creator.create_notebook()
    # elif args.fscaling:
    #     crafter =  Prepup(df)
    #     h_6 = Figlet(font='term')
    #     print(colored(h_6.renderText("\nFeature Scaling in Progress..."),'light_blue'))
    #     time.sleep(0.2)
    #     crafter.feature_scaling()
    # elif args.fmissing:
    #     crafter =  Prepup(df)
    #     h_6 = Figlet(font='term')
    #     print(colored(h_6.renderText("\nHandle Missing Data..."),'light_blue'))
    #     time.sleep(0.2)
    #     crafter.handle_missing_values()
    # elif args.nd:
    #     crafter =  Prepup(df)
    #     h_6 = Figlet(font='term')
    #     print(colored(h_6.renderText("\nCheck for Normal Distribution..."),'light_blue'))
    #     time.sleep(0.2)
    #     crafter.check_nomral_distrubution()
    # elif args.rii:
    #     missing_data = pl.read_parquet('missing_data.parquet')
    #     missing_data = missing_data.to_pandas()
    #     print(missing_data.isnull().sum())
    #     print(missing_data)
    # elif args.id:
    #     crafter =  Prepup(df)
    #     h_6 = Figlet(font='term')
    #     print(colored(h_6.renderText("\nCheck for Imabalanced Dataset..."),'light_blue'))
    #     time.sleep(0.2)
    #     crafter.imbalanced_dataset()
    # else:
    #     start = time.time() 

    #     crafter = Prepup(df)
    #     h_1 = Figlet(font='term')
    #     print(colored(h_1.renderText("Features present in the Dataset..."), 'light_blue')) 
    #     time.sleep(0.5)
    #     print(crafter.features_available())
        
    #     h_2 = Figlet(font='term')
    #     print(colored(h_2.renderText("\nDatatype of each Feature..."), 'light_blue'))
    #     time.sleep(0.5)
    #     print(crafter.dtype_features())

    #     h_3 = Figlet(font='term')
    #     print(colored(h_3.renderText("\nShape of Data..."),'light_blue'))
    #     time.sleep(0.5)
    #     print(crafter.shape_data())
                
    #     h_4 = Figlet(font='term')
    #     print(colored(h_4.renderText("\nFeatures: Missing values count..."),'light_blue'))
    #     time.sleep(0.5)
    #     crafter.missing_plot()
        
    #     h_9 = Figlet(font='term')
    #     print(colored(h_9.renderText("\nDetecting Outliers..."),'light_blue'))
    #     crafter.find_outliers()

    #     h_5 = Figlet(font='term')
    #     print(colored(h_5.renderText("\nFeature Correlation..."),'light_blue'))
    #     time.sleep(0.5)
    #     crafter.correlation_n()

    #     h_8 = Figlet(font='term')
    #     print(colored(h_8.renderText("\nExecution Time..."),'light_blue'))
    #     end_n = time.time()-start
    #     print("Operations Comepleted Successfully in {0} {1}".format(end_n,"seconds"))
                
if __name__ == '__main__':
     main()