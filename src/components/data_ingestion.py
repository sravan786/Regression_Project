import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation


# Initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')


# Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts')
        try:
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Proceeding to train test split')
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')
            return train_set, test_set
        

        except Exception as e:
            logging.info('Exception occured during data ingestion stage')
            raise CustomException(e,sys)

# run data ingestion
if __name__ == '__main__':
    data = DataIngestion()
    train_data, test_data = data.initiate_data_ingestion()
    data_path = DataIngestionConfig()
    data_transformation = DataTransformation()
    train_arr, test_arr, file_path = data_transformation.initiate_data_transformation(data_path.train_data_path, data_path.test_data_path)
