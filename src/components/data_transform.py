import numpy as np
import pandas as pd
import os
import sys
sys.path.append(".")
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransfromconfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransform:
    def __init__(self):
        self.data_transform_config=DataTransfromconfig()
    
    def get_data_transfromer_obj(self):
        try:
            numerical_columns=['writing_score',"reading_score"]
            categorical_column=[
                "race_ethnicity",
                'gender',
                "parent_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ])
            logging.info("numerical tansform completed")
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mode")),
                ("one_hot_encoding",OneHotEncoder()),
                ("standard_scaler",StandardScaler())
                ])
            logging.info("Categorical tansform completed")

            preprocessor=ColumnTransformer=([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_column)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initate_data_transform(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data sucessful")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transfromer_obj()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
    
