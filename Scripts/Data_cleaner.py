import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class data_cleaner:
    class DataCleaner:
    def drop_duplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        drop duplicate rows
        """
        data.drop_duplicates(inplace=True)

        return data
    def convert_to_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        convert column to datetime
        """

        data[['start','end']] = data[['start','end']].apply(pd.to_datetime)

        return data

    def convert_to_string(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        convert columns to string
        """
        data[['bearer_id', 'imsi', 'msisdn/number', 'imei','handset_type']] = data[['bearer_id', 'imsi', 'msisdn/number', 'imei','handset_type']].astype(str)

        return data

    def remove_whitespace_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        remove whitespace from columns
        """
        data.columns = [column.replace(' ', '_').lower() for column in data.columns]

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

    def min_max_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        """
        minmax_scaler = MinMaxScaler()
        return pd.DataFrame(minmax_scaler.fit_transform(data[self.get_numerical_columns(data)]), columns=self.get_numerical_columns(data))

    def standard_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        scale numerical columns
        """
        standard_scaler = StandardScaler()
        return pd.DataFrame(standard_scaler.fit_transform(data[self.get_numerical_columns(data)]), columns=self.get_numerical_columns(data))

    def handle_outliers(self, data:pd.DataFrame, col:str, method:str ='IQR') -> pd.DataFrame:
        """
        Handle Outliers of a specified column using Turkey's IQR method
        """
        data = data.copy()
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        
        lower_bound = q1 - ((1.5) * (q3 - q1))
        upper_bound = q3 + ((1.5) * (q3 - q1))
        if method == 'mode':
            data[col] = np.where(data[col] < lower_bound, data[col].mode()[0], data[col])
            data[col] = np.where(data[col] > upper_bound, data[col].mode()[0], data[col])
        
        elif method == 'median':
            data[col] = np.where(data[col] < lower_bound, data[col].median, data[col])
            data[col] = np.where(data[col] > upper_bound, data[col].median, data[col])
        else:
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        return data