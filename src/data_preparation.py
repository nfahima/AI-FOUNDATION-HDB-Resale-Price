# Standard library imports
import logging
import re
from typing import Any, Dict, Optional


# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

class DataPreparation:
    """
    Class used to clean and preprocess the HSB resale price data 

    Attributes:
    - config: Dict[str, Any]
      Configuration dictionary containing parameters for data cleaning
    - preprocessor: sklearn.compose.ColumnTransformer
      a preprocessor pipeline for transforming numerical, norminal and ordinal features 
    """

    def __init__(self, config: Dict[str, Any]):
        """
        initilises the DataPreparation class with configuiration dictionary

        Args:
        config (Dict[str, Any]): COnfiguratiuon dic containing paramters
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input Datafram by perfroming several preprocessing steps

        Args:
        df(pd.DataFrame): the input Dataframe containg raw data
        """
        logging.info("Starting data cleaning on %d rows.", len(df))

        # drop duplicates
        df.drop_duplicates(inplace=True)

        # Replace 'FOUR ROOM' with '4 ROOM' in the 'flat_type' column
        df['flat_type'] = df['flat_type'].replace('FOUR ROOM', '4 ROOM')

        # convert lease commece data to positive values
        df['lease_commence_date'] = df['lease_commence_date'].abs()

        # Use the .apply() method to convert the 'storey_range' column with the function
        df['storey_range'] = df['storey_range'].apply(self._convert_storey_range)

        # Fill missing values in 'town_name' column
        df = self._fill_missing_names(df=df, id_column='town_id', name_column='town_name')

        # Fill missing values in 'flatm_name' column
        df = self._fill_missing_names(df=df, id_column='flatm_id', name_column='flatm_name')

        # Drop irrelevant features
        df = df.drop(columns=['id', 'town_id', 'flatm_id'])
        n_before = len(df)
        df = df.dropna(subset=["town_name", "flatm_name"])
        if len(df)< n_before:
            logger.warning(
                "Dropped &d rows with uncoverable missing ids and name of town and flat model", 
                n_before - len(df),
            )
        # Convert 'month' column to datetime
        df['year_month'] = pd.to_datetime(df['month'], format='%Y-%m')

        # Extract year and month from the datetime object
        df['year'] = df['year_month'].dt.year
        df['month'] = df['year_month'].dt.month

        # drop year_month
        df = df.drop(columns=['year_month'])

        # Extract lease information
        df['remaining_lease_months'] = df['remaining_lease'].apply(self._extract_lease_info)

        # Drop the 'remaining_lease','block' and 'street_name' columns 
        df.drop(columns=['remaining_lease', 'block', 'street_name'], inplace=True)

        logging.info("Data cleaning completed. %d rows remain.", len(df))
        return df




    @staticmethod
    # Function to convert storey_range to ordinal scale by taking the average of the range
    def _convert_storey_range(storey_range: str) -> float:
        """
        Converts a storey range string into its average numerical value.
        
        The function takes a storey range in the format 'XX TO YY', splits it into two parts,
        converts these parts to integers, and returns the average of these integers.
        
        Args:
            storey_range (str): A string representing a range of storeys, in the format 'XX TO YY'.
            
        Returns:
            float: The average value of the two storeys in the range.
            
        Example:
            convert_storey_range('07 TO 09') -> 8.0
        """
        low, high = storey_range.split(' TO ')
        return (int(low) + int(high)) / 2


    @staticmethod
    def _fill_missing_names(df: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
        """
        Fills missing values in the 'name_column' using the 'id_column'.

        Args:
            df (pd.DataFrame): The DataFrame containing the columns to be filled.
            id_column (str): The name of the column containing the IDs.
            name_column (str): The name of the column containing the names to be filled.

        Returns:
            pd.DataFrame: The DataFrame with missing values in 'name_column' filled.
        """
        # Identify missing values in the 'name_column'
        missing_names = df[name_column].isna()

        # Create a dictionary mapping 'id_column' to 'name_column'
        name_mapping = df[[id_column, name_column]].dropna().drop_duplicates().set_index(id_column)[name_column].to_dict()

        # Fill missing 'name_column' using the mapping
        df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(name_mapping)

        return df


    @staticmethod
    def _extract_lease_info(lease_str: str) -> int:
        """
        Convert lease information from a string format to total months.

        This function takes a string representing the remaining lease period, which
        may include years and months in various formats (e.g., "70 years 3 months",
        "85 years", "67"), and converts it to the total number of months.

        Args:
            lease_str (str): The remaining lease period as a string.

        Returns:
            int: The total number of months, or None if the input is NaN.
        """

        if pd.isna(lease_str):
            return None
        
        # Regular expression to extract years and months
        lease_str = str(lease_str)
        years_match = re.search(r'(\d+)\s*years?', lease_str)
        months_match = re.search(r'(\d+)\s*months?', lease_str)
        number_match = re.match(r'^\d+$', lease_str.strip())
        
        if years_match:
            years = int(years_match.group(1))
        elif number_match:  # If only a number is present, assume it's in years
            years = int(number_match.group(0))
        else:
            years = 0
            
        months = int(months_match.group(1)) if months_match else 0
        
        # Convert the total lease period to months
        total_months = years * 12 + months
        return total_months
    
    def _create_preprocessor(self) -> ColumnTransformer:

        # Create a numerical transformer pipeline
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Create a nominal transformer pipeline
        nominal_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create an ordinal transformer pipeline
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', 
            OrdinalEncoder(categories=[self.config["flat_type_categories"]], 
                           handle_unknown='use_encoded_value', 
                           unknown_value=-1,))
        ])



        # Combine transformers into a single ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.config["numerical_features"]),
                ('nom', nominal_transformer, self.config["nominal_features"]),
                ('ord', ordinal_transformer, self.config["ordinal_features"]),
                ('pass', 'passthrough', self.config["passthrough_features"]) # Pass through the storey_range feature without transformation
            ],
            remainder='passthrough',
            n_jobs=-1
            )

        return preprocessor