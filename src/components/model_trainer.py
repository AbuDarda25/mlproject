
import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.linear_model import Ridge, Lasso, ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]            
                )
            
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                # "Linear Regression":LinearRegression(),
                "KNN Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "Adaboost Regressor":AdaBoostRegressor(),
                 # 🔥 Linear family (correct way)
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet()
            }

            # For Hyperparameter Tuning
            # params = {
            #     "Decision Tree":{
            #         'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
            #         # 'splitter':['best','random'],
            #         # 'max_features':['sqrt','log2']
            #     },
            #     "Random Forest":{
            #         # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
            #         # 'max_features':['sqrt','log2',None],
            #         'n_estimators':[8,16,32,64,128,256]

            #     },
            #     "Gradient Boosting":{
            #         # 'loss':['squared_error','huber','abosolute_error','quantile'],
            #         'learning_rate':[.1,.01,.05,.001],
            #         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            #         # 'criterion':['squared_error','friedman_mse'],
            #         # 'max_features':['auto','sqrt','log2'],
            #         'n_estimators':[8,16,32,64,128,256]
            #     },
            #     "Linear Regression":{},
            #     "KNN Regressor":{
            #         'n_neighbors':[5,7,9,11],
            #         # 'weighs':['uniform','distance'],
            #         # 'algorithm':['ball_tree','kd_tree','brute']
            #     },
            #     "XGBRegressor":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators':[8,16,32,64,128,256]

            #     },
            #     "CatBoosting Regressor":{
            #         'depth':[6,8,10],
            #         'learning_rate':[.1,.01,.05,.001],
            #         'n_estimators':[8,16,32,64,128,256]
            #     },
            #     "Adaboost Regressor":{
            #         'learning_rate':[.1,.01,.05,.001],
            #         # 'loss':['squared_error','huber','abosolute_error','quantile'],
            #         'n_estimators':[8,16,32,64,128,256]

            #     }
            # }

            # # For better Tuning I'm adding more parameters 
            # params = {
            #     "Decision Tree": {
            #         "criterion": ["squared_error", "friedman_mse"],
            #         "max_depth": [None, 5, 10, 20, 30],
            #         "min_samples_split": [2, 5, 10, 20],
            #         "min_samples_leaf": [1, 2, 5, 10],
            #         "max_features": [None, "sqrt", "log2"]
            #     },

            #     "Random Forest": {
            #         "n_estimators": [100, 200, 300],
            #         "max_depth": [None, 10, 20, 30],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 4],
            #         "max_features": ["sqrt", "log2"],
            #         "bootstrap": [True, False]
            #     },

            #     "Gradient Boosting": {
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "n_estimators": [100, 200, 300],
            #         "max_depth": [3, 5, 7],
            #         "subsample": [0.6, 0.8, 1.0],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 4]
            #     },

            #     "Linear Regression":{},
            #     "KNN Regressor": {
            #         "n_neighbors": [3, 5, 7, 9, 11, 15],
            #         "weights": ["uniform", "distance"],
            #         "p": [1, 2]   # Manhattan vs Euclidean
            #     },

            #     "XGBRegressor": {
            #         "n_estimators": [200, 300, 500],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "max_depth": [3, 5, 7],
            #         "subsample": [0.6, 0.8, 1.0],
            #         "colsample_bytree": [0.6, 0.8, 1.0],
            #         "gamma": [0, 0.1, 0.3]
            #     },

            #     "CatBoosting Regressor": {
            #         "depth": [6, 8, 10],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "iterations": [300, 500, 800],
            #         "l2_leaf_reg": [1, 3, 5, 7],
            #         "bagging_temperature": [0, 1, 3]
            #     },

            #     "Adaboost Regressor": {
            #         "n_estimators": [50, 100, 200, 300],
            #         "learning_rate": [0.01, 0.05, 0.1, 1.0],
            #         "loss": ["linear", "square", "exponential"]
            #     }

            # }

            #  From Above Tuning , I get Linear Regresion as best model , 
            # So I'm adding ridge , lasso , elasting net to remove overfitting and 
            # and make it more better
            params = {
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators':[8,16,32,64,128,256]

                },
                "Gradient Boosting":{
                    # 'loss':['squared_error','huber','abosolute_error','quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error','friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "KNN Regressor":{
                    'n_neighbors':[5,7,9,11],
                    # 'weighs':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]

                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    # 'loss':['squared_error','huber','abosolute_error','quantile'],
                    'n_estimators':[8,16,32,64,128,256]

                },
                "Ridge": {
                    "alpha": [0.01, 0.1, 1, 10, 100]
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1],
                    "max_iter": [2000, 5000]
                },
                "ElasticNet": {
                    "alpha": [0.001, 0.01, 0.1],
                    "l1_ratio": [0.3, 0.5, 0.7],
                    "max_iter": [2000, 5000]
                }

            }



            model_report:dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            ## To get the best model from dict
            best_model_name= max(
                model_report,
                key=lambda x:model_report[x]["score"]
            )

            ## To get best model name from dict
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]

            
            # Train the best model
            # best_model.fit(X_train,y_train)

            # if best model score is less than 60% then it is not best model in actual so raise an exception
            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model:{best_model_name} with R2 score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)
            return r2
            

        except Exception as e:
            raise CustomException(e,sys)
        










