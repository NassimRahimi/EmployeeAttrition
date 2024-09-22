import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import warnings
import json
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to load configuration settings from a JSON file
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_employee_data(df):
    """
    Preprocesses the employee attrition data by applying necessary data type conversions
    and resetting the Employee_ID column.

    Args:
    - df (pd.DataFrame): The raw employee attrition dataframe.

    Returns:
    - pd.DataFrame: The preprocessed dataframe.

    Raises:
    - ValueError: If any of the expected columns are missing or cannot be converted.
    """

    try:
        # Reset index to use it as Employee_ID and adjust it to start from 1
        logging.info("Resetting index and setting Employee_ID.")
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Employee_ID'}, inplace=True)
        df['Employee_ID'] = df['Employee_ID'] + 1
        df['Employee_ID'] = df['Employee_ID'].astype(int)

        # Convert satisfaction levels and evaluations to floats
        logging.info("Converting satisfaction_level and last_evaluation to float.")
        df['satisfaction_level'] = df['satisfaction_level'].astype(float)
        df['last_evaluation'] = df['last_evaluation'].astype(float)

        # Convert counts and hours to integers
        logging.info("Converting project counts, hours, and time spent to integers.")
        df['number_project'] = df['number_project'].astype(int)
        df['average_monthly_hours'] = df['average_monthly_hours'].astype(int)
        df['time_spend_company'] = df['time_spend_company'].astype(int)

        # Convert binary columns to integers
        logging.info("Converting binary columns to integers.")
        df['work_accident'] = df['work_accident'].astype(int)
        df['left'] = df['left'].astype(int)
        df['promotion_last_5years'] = df['promotion_last_5years'].astype(int)

        # Convert categorical columns to 'category' data type
        logging.info("Converting department and salary columns to 'category' type.")
        df['department'] = df['department'].astype('category')
        df['salary'] = df['salary'].astype('category')

        # Convert encoded columns to integers
        #logging.info("Converting salary_encoded and department_encoded to integers.")
        #df['salary_encoded'] = df['salary_encoded'].astype(int)
        #df['department_encoded'] = df['department_encoded'].astype(int)

        logging.info("Data preprocessing completed successfully.")
        return df

    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        raise ValueError(f"Missing expected column: {e}")

    except ValueError as e:
        logging.error(f"Data type conversion error: {e}")
        raise ValueError(f"Data type conversion error: {e}")

    except Exception as e:
        logging.error(f"Unexpected error during preprocessing: {e}")
        raise RuntimeError(f"Unexpected error during preprocessing: {e}")



def remove_outliers(data, z_score_threshold=3, numerical_columns=None):
    """
    Removes outliers from the DataFrame or NumPy array based on a Z-score threshold.

    Args:
    - data (pd.DataFrame or np.ndarray): The input data from which to remove outliers.
    - z_score_threshold (float): The Z-score threshold for outlier detection (default=3).
    - numerical_columns (list): The list of numerical columns on which to perform outlier removal.

    Returns:
    - pd.DataFrame or np.ndarray: A new DataFrame or NumPy array with outliers removed.

    Raises:
    - ValueError: If the specified columns are not found in the DataFrame.
    """
    try:
        if isinstance(data, pd.DataFrame):
            if numerical_columns is None:
                numerical_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                                     'average_monthly_hours', 'time_spend_company', 'work_accident',
                                     'promotion_last_5years']

            # Check which columns are present in the dataframe
            available_columns = [col for col in numerical_columns if col in data.columns]

            if not available_columns:
                raise ValueError(f"None of the specified numerical columns are available in the dataset: {numerical_columns}")

            logging.info(f"Processing available columns for outlier removal: {available_columns}")

            # Calculating Z-scores for the available numerical columns
            z_scores = data[available_columns].apply(zscore)

            # Identifying rows where any column's Z-score exceeds the threshold
            outliers = (z_scores.abs() > z_score_threshold).any(axis=1)

            # Creating a new dataframe without outliers
            data_cleaned = data[~outliers]
            logging.info(f"Outliers removed using Z-score threshold: {z_score_threshold}")
            logging.info(f"Original data size: {data.shape[0]}, Cleaned data size: {data_cleaned.shape[0]}")

            return data_cleaned

        elif isinstance(data, np.ndarray):
            # When dealing with a NumPy array, apply Z-score calculations directly
            z_scores = np.apply_along_axis(zscore, 0, data)

            # Identifying rows where any column's Z-score exceeds the threshold
            outliers = np.any(np.abs(z_scores) > z_score_threshold, axis=1)

            # Returning a NumPy array without outliers
            data_cleaned = data[~outliers]
            logging.info(f"Outliers removed using Z-score threshold: {z_score_threshold}")
            logging.info(f"Original data size: {data.shape[0]}, Cleaned data size: {data_cleaned.shape[0]}")

            return data_cleaned

        else:
            raise TypeError("Input data should be a pandas DataFrame or NumPy array.")

    except Exception as e:
        logging.error(f"Error removing outliers: {e}")
        raise
