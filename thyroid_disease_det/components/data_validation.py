import json
import sys

import numpy as np
import pandas as pd

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from pandas import DataFrame

from thyroid_disease_det.exception import thyroid_disease_detException
from thyroid_disease_det.logger import logging
from thyroid_disease_det.utils.main_utils import read_yaml_file, write_yaml_file
from thyroid_disease_det.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from thyroid_disease_det.entity.config_entity import DataValidationConfig
from thyroid_disease_det.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: Configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise thyroid_disease_detException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """Validates the number of columns"""
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Are all required columns present? {status}")
            return status
        except Exception as e:
            raise thyroid_disease_detException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """Validates the existence of numerical and categorical columns"""
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_categorical_columns or missing_numerical_columns)
        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """Reads dataset from CSV"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise thyroid_disease_detException(e, sys)

    def replace_invalid_values(self, df: DataFrame) -> DataFrame:
        """
        Replaces placeholder values (e.g., '?') with predefined sentinel values 
        to allow processing in Evidently for data drift detection.
        """
        try:
            logging.info("Replacing invalid values with placeholders...")
        
        # Define replacement values
            sentinel_value_num = -999  # A large negative value unlikely to be real data
            sentinel_value_cat = "missing_value"

            df.replace('?', np.nan, inplace=True)  # Standard replacement with NaN first

        # Replace NaNs in numerical columns with sentinel value
            for col in self._schema_config["numerical_columns"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric dtype
                df[col].fillna(sentinel_value_num, inplace=True)

        # Replace NaNs in categorical columns with sentinel value
            for col in self._schema_config["categorical_columns"]:
                df[col].fillna(sentinel_value_cat, inplace=True)

            logging.info("Invalid values replaced with sentinel placeholders.")
            return df
        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e

    

    


    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """Detects dataset drift using Evidently AI"""
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e

    

            

            

            

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            
            
            
            # Replace '?' with NaN before imputation
            train_df = self.replace_invalid_values(train_df)
            test_df = self.replace_invalid_values(test_df)
            
            
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_number_of_columns(dataframe=test_df)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_column_exist(df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected before transformation"
                else:
                    validation_error_msg = "Drift not detected before transformation"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")
                

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e