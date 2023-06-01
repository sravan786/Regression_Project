from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    data = DataIngestionConfig()
    train_data_path, test_data_path = data.train_data_path, data.test_data_path
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)