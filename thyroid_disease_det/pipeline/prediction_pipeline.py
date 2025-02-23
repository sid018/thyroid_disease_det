import os
import sys

import numpy as np
import pandas as pd
from thyroid_disease_det.entity.config_entity import thyroid_disease_detPredictorConfig
from thyroid_disease_det.entity.s3_estimator import thyroid_disease_detEstimator
from thyroid_disease_det.exception import thyroid_disease_detException
from thyroid_disease_det.logger import logging
from thyroid_disease_det.utils.main_utils import read_yaml_file
from pandas import DataFrame


class thyroid_disease_detData:
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                company_age
                ):
        """
        thyroid_disease_det Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age


        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e

    def get_thyroid_disease_det_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from thyroid_disease_detData class input
        """
        try:
            
            thyroid_disease_det_input_dict = self.get_thyroid_disease_det_data_as_dict()
            return DataFrame(thyroid_disease_det_input_dict)
        
        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e


    def get_thyroid_disease_det_data_as_dict(self):
        """
        This function returns a dictionary from thyroid_disease_detData class input 
        """
        logging.info("Entered get_thyroid_disease_det_data_as_dict method as thyroid_disease_detData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created thyroid_disease_det data dict")

            logging.info("Exited get_thyroid_disease_det_data_as_dict method as thyroid_disease_detData class")

            return input_data

        except Exception as e:
            raise thyroid_disease_detException(e, sys) from e

class thyroid_disease_detClassifier:
    def __init__(self,prediction_pipeline_config: thyroid_disease_detPredictorConfig = thyroid_disease_detPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise thyroid_disease_detException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of thyroid_disease_detClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of thyroid_disease_detClassifier class")
            model = thyroid_disease_detEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise thyroid_disease_detException(e, sys)