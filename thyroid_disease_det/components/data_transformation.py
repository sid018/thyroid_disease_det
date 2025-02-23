import json
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer  # ✅ Missing Import Fixed

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

from thyroid_disease_det.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from thyroid_disease_det.entity.config_entity import DataTransformationConfig
from thyroid_disease_det.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from thyroid_disease_det.exception import thyroid_disease_detException
from thyroid_disease_det.logger import logging
from thyroid_disease_det.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns,write_yaml_file
from thyroid_disease_det.entity.estimator import TargetValueMapping

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise thyroid_disease_detException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise thyroid_disease_detException(e, sys)



    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces '?' values with NaN, imputes missing categorical values with mode,
        and applies KNNImputer for numerical columns.
        """
        try:
            logging.info("Handling missing values in dataset...")

            # Replace '?' with NaN
            for column in df.columns:
                count = df[column][df[column] == '?'].count()
                if count != 0:
                    df[column] = df[column].replace('?', np.nan)

            # Fill categorical missing values with mode
            categorical_columns = self._schema_config["gender"]
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])

            # Apply KNN Imputer for numerical columns
            numerical_columns = self._schema_config["numerical_columns"]
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

            logging.info("Missing value handling complete.")
            return df

        except Exception as e:
            raise thyroid_disease_detException(e, sys)

        

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """Detects dataset drift using Evidently AI"""
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=self.data_transformation_config.drift_report_file_path, content=json_report)

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e
   
   
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore') 
            ordinal_encoder = OrdinalEncoder()


            

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            continuous_columns = self._schema_config['continuous_columns']
            #num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            continuous_pipeline = Pipeline(steps=[
                                ('power_transform', PowerTransformer(method='yeo-johnson')),
                                ('standard_scaler', StandardScaler())
                                    ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns+ ['sex']),
                    ("Continuous_Transformer", continuous_pipeline, continuous_columns)
                ],
                    remainder='passthrough'
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e

    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            drift_error_msg = ""
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

                drop_cols = self._schema_config['drop_columns']

                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)

                input_feature_train_df = self.handle_missing_values(input_feature_train_df)
                
                
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )


                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]


                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

                logging.info("drop the columns in drop_cols of Test dataset")

                input_feature_test_df = self.handle_missing_values(input_feature_test_df)

                target_feature_test_df = target_feature_test_df.replace(
                TargetValueMapping()._asdict()
                )

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                transformed_train_df = pd.DataFrame(input_feature_train_arr,columns=preprocessor.get_feature_names_out(input_feature_train_df.columns))
                transformed_test_df = pd.DataFrame(input_feature_test_arr,columns=preprocessor.get_feature_names_out(input_feature_test_df.columns))

                # ✅ Detect Drift After Transformation 
                drift_status_after = self.detect_dataset_drift(transformed_train_df, transformed_test_df)
                if drift_status_after:
                    logging.info(f"Drift detected.")
                    drift_error_msg = "Drift detected after transformation"
                else:
                    drift_error_msg = "Drift not detected after transformation"
            
                logging.info("drift_error_msg: {drift_error_msg}")
                

                


                logging.info("Applying RandomOverSampler on Training dataset")
                ros = RandomOverSampler(random_state=42, sampling_strategy='minority')

                # Apply RandomOverSampler only on the training set
                input_feature_train_final, target_feature_train_final = ros.fit_resample(input_feature_train_arr, target_feature_train_df)

                #logging.info("Applied SMOTEENN on training dataset")

                #logging.info("Applying SMOTEENN on testing dataset")

                input_feature_test_final, target_feature_test_final = input_feature_test_arr, target_feature_test_df
                

                #logging.info("Applied SMOTEENN on testing dataset")

                #logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    drift_status=drift_status_after,
                    message= drift_error_msg,
                    drift_report_file_path=self.data_transformation_config.drift_report_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e





  