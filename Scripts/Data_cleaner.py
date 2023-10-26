import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, Normalizer
from sklearn.impute import SimpleImputer

class data_cleaner:
    def drop_duplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        drop duplicate rows
        """
        data.drop_duplicates(inplace=True)

        return data
 
    def percent_missing(self, data: pd.DataFrame) -> float:
        """
        calculate the percentage of missing values from dataframe
        """
        totalCells = np.product(data.shape)
        missingCount = data.isnull().sum()
        totalMising = missingCount.sum()

        return round(totalMising / totalCells * 100, 2)

    def get_numerical_columns(self, data: pd.DataFrame) -> list:
        """
        get numerical columns
        """
        return data.select_dtypes(include=['number']).columns.to_list()

    def get_categorical_columns(self, data: pd.DataFrame) -> list:
        """
        get categorical columns
        """
        return  data.select_dtypes(include=['object','datetime64[ns]']).columns.to_list()

    def percent_missing_column(self, data: pd.DataFrame, col:str) -> float:
        """
        calculate the percentage of missing values for the specified column
        """
        try:
            col_len = len(data[col])
        except KeyError:
            print(f"{col} not found")
        missing_count = data[col].isnull().sum()

        return round(missing_count / col_len * 100, 2)
    
    def fill_missing_values_categorical(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        fill missing values with specified method
        """

        categorical_columns = data.select_dtypes(include=['object','datetime64[ns]']).columns

        if method == "ffill":

            for col in categorical_columns:
                data[col] = data[col].fillna(method='ffill')

            return data

        elif method == "bfill":

            for col in categorical_columns:
                data[col] = data[col].fillna(method='bfill')

            return data

        elif method == "mode":
            
            for col in categorical_columns:
                data[col] = data[col].fillna(data[col].mode()[0])

            return data
        else:
            print("Method unknown")
            return data

    def fill_missing_values_numeric(self, data: pd.DataFrame, method: str,columns: list =None) -> pd.DataFrame:
        """
        fill missing values with specified method
        """
        if(columns==None):
            numeric_columns = self.get_numerical_columns(data)
        else:
            numeric_columns=columns

        if method == "mean":
            for col in numeric_columns:
                data[col].fillna(data[col].mean(), inplace=True)

        elif method == "median":
            for col in numeric_columns:
                data[col].fillna(data[col].median(), inplace=True)
        else:
            print("Method unknown")
        
        return data

    def remove_nan_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        remove columns with nan values for categorical columns
        """

        categorical_columns = self.get_categorical_columns(data)
        for col in categorical_columns:
            data = data[data[col] != 'nan']

        return data

    def normalizer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        normalize numerical columns
        """
        norm = Normalizer()
        return pd.DataFrame(norm.fit_transform(data[self.get_numerical_columns(data)]), columns=self.get_numerical_columns(data))