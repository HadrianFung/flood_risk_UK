'''
This is a holistic tool that combines all the models and data processing in order
to come out with flood prediction and house price prediction, and uses this
information to assess the final risk class of a given postcode.
'''

import os
import re
import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

from flood_tool.geo import get_easting_northing_from_gps_lat_long
from flood_tool.geo import get_gps_lat_long_from_easting_northing
from .autotuner import AutoTuner

set_config(transform_output='pandas')

__all__ = [
    "Tool",
    "_data_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
    "MLP_historic_flooding",
    "KNN_historic_flooding",
    "RF_historic_flooding",
    "GBR_median_price",
    "KNR_median_price",
    "RFR_median_price",
    "KNN_local_authority",
    "SVC_local_authority",
    "RF_local_authority",
    "RF_riskLabel_from_postcode",
    "LR_riskLabel_from_postcode",
    "KNN_riskLabel_from_postcode",
    "RF_riskLabel_from_location",
    "LR_riskLabel_from_location"
]

_data_dir = os.path.join(os.path.dirname(__file__), "resources")


# classification/regression methods
flood_class_from_postcode_methods = {
    "RF_riskLabel_from_postcode": "RandomForestClassifier",
    "LR_riskLabel_from_postcode": "Logistic Regression",
    "KNN_riskLabel_from_postcode": "K-Nearest Neighbours (KNN)"
}
flood_class_from_location_methods = {
    "RF_riskLabel_from_location": "RandomForestClassifier",
    "LR_riskLabel_from_location": "Logistic Regression",
    "KNN_riskLabel_from_location": "K-Nearest Neighbours (KNN)"
}
historic_flooding_methods = {
    "MLP_historic_flooding": "Multi-layer Perceptron (MLP)",
    "RF_historic_flooding": "Random Forest (RF)",
    "KNN_historic_flooding": "K-Nearest Neighbours (KNN)",
}
house_price_methods = {
    "GBR_median_price": "GradientBoosting Regressor (GBR)",
    "KNR_median_price": "KNeighbors Regressor (KNR)",
    "RFR_median_price": "RandomForest Regressor (RFR)"
}
local_authority_methods = {
    "KNN_local_authority": "K-Nearest Neighbours (KNN)",
    "SVC_local_authority": "Support Vector Classifier (SVC)",
    "RF_local_authority": "Random Forest Classifier (RF)"
}

class MLP_historic_flooding:
    '''Class to train MLP model to predict historic flooding from easting, northing,
     soil type and elevation.'''
    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'soilType', 'elevation','historicallyFlooded']]

        num_data = ['easting', 'northing', 'elevation']
        cat_data = ['soilType']
        num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        self.pipeline = ColumnTransformer([('num',num_pipe, num_data),
                                        ('cat', cat_pipe, cat_data)])
        
        self.X = df.copy(deep=True).drop(columns=['historicallyFlooded'])
        self.y = df.copy(deep=True)['historicallyFlooded']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)  

        mlp_classifier = MLPClassifier(activation='tanh', early_stopping=True,
              max_iter=400)
        self.model = make_pipeline(self.pipeline, mlp_classifier)      

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class KNN_historic_flooding:
    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'soilType', 'elevation','historicallyFlooded']]
        num_data = ['easting', 'northing', 'elevation']
        cat_data = ['soilType']

        num_pipe = make_pipeline(SimpleImputer(), StandardScaler())
        cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                 OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        self.pipeline = ColumnTransformer([('num',num_pipe, num_data),
                                        ('cat', cat_pipe, cat_data)])
        self.X = df.copy(deep=True).drop(columns=['historicallyFlooded'])
        self.y = df.copy(deep=True)['historicallyFlooded']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        knn_classifier = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=2, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, knn_classifier)
        
    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class RF_historic_flooding:
    def __init__(self, df):
        '''Initialize the model.'''
        # Feature lists
        num_data_elevation = ['elevation']
        num_data_others = ['easting', 'northing']
        cat_data = ['soilType']

        # Log transformation for 'elevation'
        log_transformer = FunctionTransformer(np.log1p, validate=False)

        # Pipelines
        elevation_pipe = make_pipeline(SimpleImputer(strategy='median'), log_transformer)
        other_num_pipe = make_pipeline(SimpleImputer(strategy='median'))
        cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                 OneHotEncoder(handle_unknown='ignore', sparse_output=False))

        # Column transformer
        self.pipeline = ColumnTransformer([
            ('elevation', elevation_pipe, num_data_elevation),
            ('other_num', other_num_pipe, num_data_others),
            ('cat', cat_pipe, cat_data)
        ])

        # Data preparation
        self.X = df[['easting', 'northing', 'elevation', 'soilType']]
        self.y = df['historicallyFlooded']

        # Splitting the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        rf_classifier = RandomForestClassifier(max_depth=49, max_features=3, min_samples_split=4,
                       n_estimators=269, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, rf_classifier)

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train , self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class GBR_median_price:
    """Class to train GradientBoosting Regressor model to predict median price from postcode."""

    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['postcode', 'riskLabel', 'households', 'numberOfPostcodeUnits',
                 'headcount', 'catsPerHousehold', 'dogsPerHousehold', 'retailNum', 'medianPrice']]
        df = df.dropna(subset=['medianPrice'])
        num_data = df.select_dtypes(exclude=object).columns
        num_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())
        self.pipeline = ColumnTransformer([
            ('num_pipe', num_pipe, num_data)
        ])

        self.X = df.copy(deep=True).drop(columns=['medianPrice'])
        self.y = df.copy(deep=True)['medianPrice']

        # Split the train set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=42)
        num_data = self.X_train.select_dtypes(exclude=object).columns
        num_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())

        self.pipeline = ColumnTransformer([('num', num_pipe, num_data)])
        
        gbr_regressor = GradientBoostingRegressor(criterion='squared_error', 
                                                  learning_rate=0.6117795985486375,
                                                  loss='absolute_error', max_features=1.0,
                                                  n_estimators=17)

        self.model = make_pipeline(self.pipeline, gbr_regressor)

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the RMSE of the model.'''
        y_pred = self.predict(self.X_test)
        return np.sqrt(mean_squared_error(self.y_test, y_pred))

    def predict(self, X_pred):
        '''Return the predicted median price of the input data.'''
        return np.round(self.model.predict(X_pred), 2)

class KNR_median_price:
    """Class to train KNN Regressor model to predict median price from postcode."""

    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['postcode', 'riskLabel', 'households', 'numberOfPostcodeUnits',
                 'headcount', 'catsPerHousehold', 'dogsPerHousehold', 'retailNum', 'medianPrice']]
        df = df.dropna(subset=['medianPrice'])
        num_data = df.select_dtypes(exclude=object).columns
        num_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())
        self.pipeline = ColumnTransformer([
            ('num_pipe', num_pipe, num_data)
        ])

        self.X = df.copy(deep=True).drop(columns=['medianPrice'])
        self.y = df.copy(deep=True)['medianPrice']

        # Split the train set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=42)
        num_data = self.X_train.select_dtypes(exclude=object).columns
        num_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())

        self.pipeline = ColumnTransformer([('num', num_pipe, num_data)])

        knr_regressor = KNeighborsRegressor(algorithm='brute', n_neighbors=4, weights='distance', n_jobs=-1)
        self.model = make_pipeline(self.pipeline, knr_regressor)
    
    
    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the RMSE of the model.'''
        y_pred = self.predict(self.X_test)
        return np.sqrt(mean_squared_error(self.y_test, y_pred))

    def predict(self, X_pred):
        '''Return the predicted median price of the input data.'''
        return self.model.predict(X_pred)

class RFR_median_price:
    """Class to train RandomForest Regressor model to predict median price from postcode."""

    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['postcode', 'riskLabel', 'households',
                 'numberOfPostcodeUnits', 'headcount', 'catsPerHousehold', 'dogsPerHousehold', 'retailNum', 'medianPrice']]
        df = df.dropna(subset=['medianPrice'])
        num_data = df.select_dtypes(exclude=object).columns
        num_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())
        self.pipeline = ColumnTransformer([
            ('num_pipe', num_pipe, num_data)
        ])

        self.X = df.copy(deep=True).drop(columns=['medianPrice'])
        self.y = df.copy(deep=True)['medianPrice']

        # Split the train set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.8, random_state=42)
        num_data = self.X_train.select_dtypes(exclude=object).columns
        num_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())

        self.pipeline = ColumnTransformer([('num', num_pipe, num_data)])

        rf_regressor = RandomForestRegressor(max_features='log2', n_estimators=117, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, rf_regressor)
     
    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the RMSE of the model.'''
        y_pred = self.predict(self.X_test)
        return np.sqrt(mean_squared_error(self.y_test, y_pred))

    def predict(self, X_pred):
        '''Return the predicted median price of the input data.'''
        return self.model.predict(X_pred)


class KNN_local_authority:
    """Class to train KNN model to predict local authority from easting and northing."""

    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'localAuthority']]
        self.X = df.copy(deep=True).drop(columns=['localAuthority'])
        self.y = df.copy(deep=True)['localAuthority']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

        knn_classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, knn_classifier)

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted local authority of the input data.'''
        return self.model.predict(X)

class SVC_local_authority:
    """Class to train SVC model to predict local authority from easting and northing."""
    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'localAuthority']]
        self.X = df.copy(deep=True).drop(columns=['localAuthority'])
        self.y = df.copy(deep=True)['localAuthority']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
        
        svc_classifier = SVC(kernel='linear', C=2., probability=True)
        self.model = make_pipeline(self.pipeline, svc_classifier)

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })
        return metrics_df

    def predict(self, X):
        '''Return the predicted local authority of the input data.'''
        return self.model.predict(X)

class RF_local_authority:
    """Class to train RF model to predict local authority from easting and northing."""

    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'localAuthority']]
        self.X = df.copy(deep=True).drop(columns=['localAuthority'])
        self.y = df.copy(deep=True)['localAuthority']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

        rm_classifier = RandomForestClassifier(n_jobs=-1)
        self.model = make_pipeline(self.pipeline, rm_classifier)

    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self,X):
        '''Return the predicted local authority of the input data.'''
        return self.model.predict(X)


class RF_riskLabel_from_postcode:
    """Class to train random forest model to predict risk label from postcode."""

    def __init__(self, df):
        '''Initialize the model.'''
        # Feature lists
        num_data_elevation = ['elevation']  # 'elevation' to be log-transformed
        num_data_others = ['easting', 'northing']  # Other numerical data
        cat_data = ['soilType']   # Categorical data

        # Log transformation for 'elevation'
        log_transformer = FunctionTransformer(np.log1p, validate=False)

        # Pipelines
        elevation_pipe = make_pipeline(SimpleImputer(strategy='median'), log_transformer)
        other_num_pipe = make_pipeline(SimpleImputer(strategy='median'))
        cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                 OneHotEncoder(handle_unknown='ignore', sparse_output=False))

        # Column transformer
        self.pipeline = ColumnTransformer([
            ('elevation', elevation_pipe, num_data_elevation),
            ('other_num', other_num_pipe, num_data_others),
            ('cat', cat_pipe, cat_data)
        ])

        # Data preparation
        self.X = df[['easting', 'northing', 'elevation', 'soilType']]
        self.y = df['riskLabel']

        # Splitting the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)
        rf_classifier = RandomForestClassifier(max_depth=49, max_features=3, min_samples_split=4,
                                               n_estimators=269, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, rf_classifier)


    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train , self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class LR_riskLabel_from_postcode:
    """Class to train random forest model to predict risk label from postcode."""
    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'soilType', 'elevation','riskLabel']]
        num_data = ['easting', 'northing', 'elevation']
        cat_data = ['soilType']

        num_pipe = make_pipeline(SimpleImputer(), StandardScaler())
        cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                 OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        self.pipeline = ColumnTransformer([('num',num_pipe, num_data),
                                        ('cat', cat_pipe, cat_data)])
        self.X = df.copy(deep=True).drop(columns=['riskLabel'])
        self.y = df.copy(deep=True)['riskLabel']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        lr = LogisticRegression(C=1.6599452033620266, max_iter=3000, solver='liblinear')
        self.model = make_pipeline(self.pipeline, lr)


    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })
        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class KNN_riskLabel_from_postcode:
    """Class to train random forest model to predict risk label from postcode."""
    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'soilType', 'elevation','riskLabel']]
        num_data = ['easting', 'northing', 'elevation']
        cat_data = ['soilType']

        num_pipe = make_pipeline(SimpleImputer(), StandardScaler())
        cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                 OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        self.pipeline = ColumnTransformer([('num',num_pipe, num_data),
                                        ('cat', cat_pipe, cat_data)])
        self.X = df.copy(deep=True).drop(columns=['riskLabel'])
        self.y = df.copy(deep=True)['riskLabel']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        knn_classifier = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=8, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, knn_classifier) 


    def fit(self):
        '''Fit the model.'''
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class RF_riskLabel_from_location:
    """Class to train random forest model to predict risk label from easting and northing (or longtitude and latitude)."""

    def __init__(self, df):
        '''Initialize the model.'''
        df = df[['easting', 'northing', 'riskLabel']]
        self.X = df.copy(deep=True)[['easting', 'northing']]
        self.y = df.copy(deep=True)['riskLabel']

        self.pipeline = make_pipeline(SimpleImputer(strategy='median'),
                                      StandardScaler())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        rf_classifier = RandomForestClassifier(max_depth=49, max_features=3, min_samples_split=4, n_estimators=269, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, rf_classifier)

    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        return self.model.predict(X)

class LR_riskLabel_from_location:
    """Class to train logistic regression model to predict risk label from easting and northing (or longtitude and latitude)."""
    def __init__(self, df):
        df = df[['easting', 'northing', 'riskLabel']]
        X = df.copy(deep=True)[['easting', 'northing']]
        y = df.copy(deep=True)['riskLabel']

        self.pipeline = make_pipeline(SimpleImputer(strategy='median'),
                                      StandardScaler())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        lr = LogisticRegression(C=1.6599452033620266, max_iter=3000, solver='liblinear')
        self.model = make_pipeline(self.pipeline, lr)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        '''Return the accuracy, precision, recall and f1 score of the model.'''
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        '''Return the predicted risk label of the input data.'''
        return self.model.predict(X)

class KNN_riskLabel_from_location:
    """Class to train logistic regression model to predict risk label from easting and northing (or longtitude and latitude)."""
    def __init__(self, df):
        df = df[['easting', 'northing', 'riskLabel']]
        self.X = df.copy(deep=True)[['easting', 'northing']]
        self.y = df.copy(deep=True)['riskLabel']

        self.pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Balancing the dataset
        over = RandomOverSampler(random_state=42)
        self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)

        knn_classifier = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=8, n_jobs=-1)
        self.model = make_pipeline(self.pipeline, knn_classifier)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]
        })

        return metrics_df

    def predict(self, X):
        return self.model.predict(X)


class Tool():
    """Class to interact with a postcode database file."""

    def __init__(self, unlabelled_unit_data="", labelled_unit_data="",
                 sector_data="", district_data="", additional_data=None):
        """
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additiona .csv files containing addtional
            information on households.
        """
        if additional_data is None:
            additional_data = {}
        if unlabelled_unit_data == "":
            unlabelled_unit_data = os.path.join(_data_dir,
                                                'postcodes_unlabelled.csv')

        if labelled_unit_data == "":
            labelled_unit_data = os.path.join(_data_dir,
                                              'postcodes_labelled.csv')

        if sector_data == "":
            sector_data = os.path.join(_data_dir,
                                       'sector_data.csv')
        if district_data == "":
            district_data = os.path.join(_data_dir,
                                         'district_data.csv')

        self._postcodedb = pd.read_csv(unlabelled_unit_data)
        self._labelled_postcodedb = pd.read_csv(labelled_unit_data)
        self._sector_data = pd.read_csv(sector_data)
        self._district_data = pd.read_csv(district_data)

        # merge the postcode database with retail data, sector data and district data
        self._pricedb = self.mp_merge_df(self._labelled_postcodedb)

        # dictionary of model keys and corresponding class instances
        flood_class_from_postcode_models = {
            "RF_riskLabel_from_postcode": RF_riskLabel_from_postcode(df=self._labelled_postcodedb),
            "LR_riskLabel_from_postcode": LR_riskLabel_from_postcode(df=self._labelled_postcodedb),
            "KNN_riskLabel_from_postcode":  KNN_riskLabel_from_postcode(df=self._labelled_postcodedb)
        }
        flood_class_from_location_models = {
            "RF_riskLabel_from_location": RF_riskLabel_from_location(df=self._labelled_postcodedb),
            "LR_riskLabel_from_location": LR_riskLabel_from_location(df=self._labelled_postcodedb),
            "KNN_riskLabel_from_location": KNN_riskLabel_from_location(df=self._labelled_postcodedb)
        }
        historic_flooding_models = {
            "MLP_historic_flooding": MLP_historic_flooding(df=self._labelled_postcodedb),
            "RF_historic_flooding": RF_historic_flooding(df=self._labelled_postcodedb),
            "KNN_historic_flooding": KNN_historic_flooding(df=self._labelled_postcodedb),
        }
        house_price_models = {
            "GBR_median_price": GBR_median_price(df=self._pricedb),
            "KNR_median_price": KNR_median_price(df=self._pricedb),
            "RFR_median_price": RFR_median_price(df=self._pricedb)
        }
        local_authority_models = {
            "KNN_local_authority": KNN_local_authority(df=self._labelled_postcodedb),
            "SVC_local_authority": SVC_local_authority(df=self._labelled_postcodedb),
            "RF_local_authority": RF_local_authority(df=self._labelled_postcodedb)
        }

        # Start unpacking to merge dictionaries
        models_dict = {
            **flood_class_from_postcode_models,
            **flood_class_from_location_models,
            **historic_flooding_models,
            **house_price_models,
            **local_authority_models
        }
        self.models_dict = models_dict

    def mp_merge_df(self, X):
        """Merge the input postcode with retail data, sector data and district data.

        Use for the input of median house price model.
        
        Parameters
        ----------

        X : pd.DataFrame
            The input postcodes dataframe. Only a column named 'postcode'.

        Returns
        ----------

        df : pd.DataFrame
            New dataframe including other feature columns.
        """
        def _split_postcode_into_district(postcode: str):
            postcode_sub_ls = re.split(r'(\d+)', postcode)
            # in case the input postcode is A999AA, the standard format of which should be A99 9AA
            if len(postcode_sub_ls[1]) == 3:
                postcode_sub_ls[1] = postcode_sub_ls[1][:-1]
            return ''.join(postcode_sub_ls[:2])

        # process retail data
        retail_data = os.path.join(_data_dir,
                                    'retail_num_by_district.csv')
        retail_data = pd.read_csv(retail_data, index_col=0)
        retail_data['postcodeDistrict'] = retail_data['postcodeDistrict'].apply(lambda x: x.lower())

        # process sector data
        sector_data = self._sector_data.copy(deep=True)
        sector_data['postcodeSector'] = sector_data['postcodeSector'].apply(lambda x: re.sub(' +', ' ', x.lower()))

        # process district data
        district_data = os.path.join(_data_dir,
                                    'district_data.csv')
        district_data = pd.read_csv(district_data)
        district_data['postcodeDistrict'] = district_data['postcodeDistrict'].apply(lambda x: x.lower())

        labelled_data = X.copy(deep=True)

        # process the labelled data
        labelled_data['postcode'] = labelled_data['postcode'].apply(lambda x: re.sub(' +', ' ', x.lower()))

        labelled_data['postcodeSector'] = labelled_data['postcode'].apply(lambda x: x[:-2])
        labelled_data['postcodeDistrict'] = labelled_data['postcode'].apply(_split_postcode_into_district)

        df = pd.merge(labelled_data, sector_data, on=['postcodeSector'], how='left')
        df = pd.merge(df, district_data, on=['postcodeDistrict'], how='left')
        df = pd.merge(df, retail_data, on=['postcodeDistrict'], how='left')
        df.drop(columns=['postcodeSector', 'postcodeDistrict'], inplace=True)

        return df

    def train(self, models=[], update_labels="", tune_hyperparameters=False, ensemble=None):
        """Train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to train
        update_labels : str, optional
            Filename of a .csv file containing a labelled set of samples.
        tune_hyperparameters : bool, optional
            If true, models can tune their hyperparameters, where
            possible. If false, models use your chosen default hyperparameters.
        ensemble : None or str (optional)
            'hard' or 'soft' voting ensemble method on all models being trained. 
            'reg' should be specified for voting ensembles of regression models.
            If None, no ensemble method is used and will take in the method as specified.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.train(fcp_methods[0])  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        """

        if update_labels:
            print("updating labelled sample file")
            self._labelled_postcodedb = pd.read_csv(update_labels)

        # if type(models) != list:
        #     models = [models]
        if not isinstance(models, list):
            models = [models]

        task = 'classification'

        for model in models:
            if tune_hyperparameters:
                print(f"tuning {model} hyperparameters")
                # Change the model in the dictionary to the tuned model directly
                self.models_dict[model].model = AutoTuner(X = self.models_dict[model].X,
                                                          Y = self.models_dict[model].y,
                                                          pipeline = self.models_dict[model].model
                                                          ).auto_tuning()

            #if type(ensemble) != str:
            if isinstance(ensemble, str):
                print(f"training {model}")
                self.models_dict[model].fit()
            else:
                if model in house_price_methods:
                    task = 'regression'
            
        #if type(ensemble) == str:
        if isinstance(ensemble, str):
            if task == 'classification':
                voting_ensemble = VotingClassifier(
                    estimators=[(model, self.models_dict[model].model) for model in models],
                    voting=ensemble,
                    n_jobs=-1
                    )
            elif task == 'regression' or ensemble == 'reg':
                voting_ensemble = VotingRegressor(
                    estimators=[(model, self.models_dict[model].model) for model in models],
                    n_jobs=-1
                    )
                
            voting_ensemble.fit(X = self.models_dict[models[0]].X,
                                y = self.models_dict[models[0]].y)
                
            self.ensemble = voting_ensemble
        

    def get_feature_from_postcode(self, postcodes):
        """
        Retrieve a DataFrame containing the OSGB36 eastings, northings, soil type, 
        and elevation for a given set of postcodes.

        This function accesses an internal postcode database and returns selected 
        geographical and environmental features for each postcode in the input list.
        Postcodes not found in the database will have their corresponding rows filled
        with NaN values.

        Parameters
        ----------
        postcodes : sequence of strs
            A list or sequence of postcodes for which features are to be retrieved.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by the input postcodes, containing columns for OSGB36 
            eastings, northings, soil type, and elevation. Rows corresponding to 
            invalid postcodes (not present in the database) will contain NaN values.
        """
        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing", "soilType", "elevation"]]

    def lookup_easting_northing(self, postcodes):
        """Get a dataframe of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of OSGB36 easthing and northing,
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the input unlabelled postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['M34 7QL'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                  easting  northing
        postcode
        M34 7QL    393470	 394371
        >>> results = tool.lookup_easting_northing(['M34 7QL', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                  easting  northing
        postcode
        M34 7QL  393470.0  394371.0
        AB1 2PQ       NaN       NaN
        """

        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing"]]

    def lookup_lat_long(self, postcodes):
        """Get a Pandas dataframe containing GPS latitude and longitude
        information for a collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NAN.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        """
        df=self.lookup_easting_northing(postcodes)
        east=df['easting'].values.astype(np.float64)
        north=df['northing'].values.astype(np.float64)

        lat, long=get_gps_lat_long_from_easting_northing(east, north)
        return pd.DataFrame({'latitude':lat, 'longitude':long},
                            index=postcodes)

    def predict_flood_class_from_postcode(self, postcodes, method="RF_riskLabel_from_postcode"):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.
            If 'ensemble' is specified as the method, the ensemble model trained will be used.



        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """
        if method == 'ensemble':
            features = self.get_feature_from_postcode(postcodes)
            predictions = self.ensemble.predict(features)
            return pd.Series(
                data=predictions,
                index=(postcode for postcode in postcodes),
                name="riskLabel",
            )
        elif method not in self.models_dict.keys():
            raise NotImplementedError(f"method {method} not implemented")
        features = self.get_feature_from_postcode(postcodes)
        predictions=self.models_dict[method].predict(features)
        return pd.Series(
            data=predictions,
            index=(postcode for postcode in postcodes),
            name="riskLabel",
        )

    def predict_flood_class_from_OSGB36_location(
        self, eastings, northings, method="RF_riskLabel_from_location"
    ):
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.
            If 'ensemble' is specified as the method, the ensemble model trained will be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        """
        if method == 'ensemble':
            features = pd.DataFrame({'easting': eastings, 'northing': northings})
            predictions = self.ensemble.predict(features)
            return pd.Series(
                data=predictions,
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
            )
        elif method not in self.models_dict.keys():
            raise NotImplementedError(f"method {method} not implemented")

        features = pd.DataFrame({'easting': eastings, 'northing': northings})
        predictions=self.models_dict[method].predict(features)
        return pd.Series(
            data=predictions,
            index=((est, nth) for est, nth in zip(eastings, northings)),
            name="riskLabel",
        )

    def predict_flood_class_from_WGS84_locations(
        self, longitudes, latitudes, method="RF_riskLabel_from_location"
    ):
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : str (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.
            If 'ensemble' is specified as the method, the ensemble model trained will be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations.
        """

        if method == 'ensemble':
            eastings, northings=get_easting_northing_from_gps_lat_long(
                longitudes, latitudes, rads=False
            )
            features = pd.DataFrame({'easting': eastings, 'northing': northings})
            predictions = self.ensemble.predict(features)
            return pd.Series(
                data=predictions,
                #index=[(lng, lat) for lng, lat in zip(longitudes, latitudes)],
                index = list(zip(longitudes, latitudes)),
                name="riskLabel",
            )
        if method not in self.models_dict.keys():
            raise NotImplementedError(f"method {method} not implemented")

        eastings, northings=get_easting_northing_from_gps_lat_long(
            longitudes, latitudes, rads=False
        )
        features = pd.DataFrame({'easting': eastings, 'northing': northings})
        predictions=self.models_dict[method].predict(features)
        return pd.Series(
            data=predictions,
            #index=[(lng, lat) for lng, lat in zip(longitudes, latitudes)],
            index = list(zip(longitudes, latitudes)),
            name="riskLabel",
        )

    def predict_median_house_price(
        self, postcodes, method="GBR_median_price"
    ):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.
            If 'ensemble' is specified as the method, the ensemble model trained will be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """
        if method == 'ensemble':
            self.train(['RF_riskLabel_from_postcode'])
            risk_label = self.predict_flood_class_from_postcode(postcodes, method='RF_riskLabel_from_postcode')

            features = pd.DataFrame({'postcode': postcodes, 'riskLabel': risk_label})
            features = self.mp_merge_df(features)

            predictions = self.ensemble.predict(features)
            return pd.Series(
                data=predictions,
                index=(postcode for postcode in postcodes),
                name='medianPrice'
            )
        if method in ['KNR_median_price', 'GBR_median_price', 'RFR_median_price']:

            # use the risk label model to predict the riskLabel first
            # then add it to the features and merge with other datasets
            self.train(['RF_riskLabel_from_postcode'])
            risk_label = self.predict_flood_class_from_postcode(postcodes, method='RF_riskLabel_from_postcode')

            features = pd.DataFrame({'postcode': postcodes, 'riskLabel': risk_label})
            features = self.mp_merge_df(features)

            predictions = self.models_dict[method].predict(features)

            return pd.Series(
                data=predictions,
                index=(postcode for postcode in postcodes),
                name='medianPrice'
            )

        raise NotImplementedError(f"method {method} not implemented")

    def predict_local_authority(
        self, eastings, northings, method="KNN"
    ):
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.
            If 'ensemble' is specified as the method, the ensemble model trained will be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            postcodes, and indexed by postcodes.
        """  
        if method == 'ensemble':
            features = pd.DataFrame({'easting': eastings, 'northing': northings})
            predictions = self.ensemble.predict(features)
            return pd.Series(
                data=predictions,
                index=[(easting, northing) for easting, northing in zip(eastings, northings)],
                name="localAuthority",
            )      
        if method not in self.models_dict.keys():
            raise NotImplementedError(f"method {method} not implemented")

        features = pd.DataFrame({'easting': eastings, 'northing': northings})
        predictions=self.models_dict[method].predict(features)
        return pd.Series(
            data=predictions,
            #index=[(easting, northing) for easting, northing in zip(eastings, northings)],
            index = list(zip(eastings, northings)),
            name="localAuthority",
        )

    def predict_historic_flooding(
        self, postcodes, method="MLP_historic_flooding"
    ):
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.
            If 'ensemble' is specified as the method, the ensemble model trained will be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        """
        if method == 'ensemble':
            features = self.get_feature_from_postcode(postcodes)
            predictions = self.ensemble.predict(features)
            return pd.Series(
                data=predictions,
                index=(postcode for postcode in postcodes),
                name="historicallyFlooded",
            )
        if method not in self.models_dict.keys():
            raise NotImplementedError(f"method {method} not implemented")

        features = self.get_feature_from_postcode(postcodes)
        predictions=self.models_dict[method].predict(features)
        return pd.Series(
            data=predictions,
            index=(postcode for postcode in postcodes),
            name="historicallyFlooded",
        )

    def predict_total_value(self, postal_data):
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcodesectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        # transform post code to postcodeSector so that we can look up households in sector_data.csv
        if len(postal_data[0])>6: #when the input is postcode units
            transformed_postcodes = set()

            for postcode in postal_data:
                trimmed_postcode = postcode[:-2]
                trimmed_postcode = trimmed_postcode.replace(" ", "")

                if len(trimmed_postcode) == 4:
                    formatted_postcode = trimmed_postcode[:-1] + "  " + trimmed_postcode[-1]
                else:
                    formatted_postcode = trimmed_postcode[:-1] + " " + trimmed_postcode[-1]

                transformed_postcodes.add(formatted_postcode)

            # Look up for households and store them in the series
            households_values = []

            for postcode_sector in transformed_postcodes:
                matching_rows = self._sector_data[self._sector_data['postcodeSector'] == postcode_sector]
                if matching_rows.empty:
                    raise ValueError(f"No matching data for {postcode_sector}.")
                households_values.append(matching_rows['households'].iloc[0])

            households_series = pd.Series(households_values, index=transformed_postcodes)

            # get median house price and store them in the series
            median_house_price_values = self.predict_median_house_price(postal_data)
            
            median_house_price_series = pd.Series(data=median_house_price_values.values, index=transformed_postcodes)

            if len(households_series) != len(median_house_price_series):
                raise ValueError("Series lengths do not match.")

            return median_house_price_series * households_series

        else: #when the input is postcodesector
            formatted_postcodesectors = []
            for postcode_sector in postal_data:
                trimmed_postcode = postcode_sector.replace(" ", "")
                if len(trimmed_postcode) == 4:
                    formatted_postcodesector = trimmed_postcode[:-1] + "  " + trimmed_postcode[-1]
                else:
                    formatted_postcodesector = trimmed_postcode[:-1] + " " + trimmed_postcode[-1]

                formatted_postcodesectors.append(formatted_postcodesector)

            # Look up for households and store them in the series
            households_values = []

            for postcode_sector in formatted_postcodesectors:
                matching_rows = self._sector_data[self._sector_data['postcodeSector'] == postcode_sector]
                if matching_rows.empty:
                    raise ValueError(f"No matching data for {postcode_sector}.")
                households_values.append(matching_rows['households'].iloc[0])

            households_series = pd.Series(households_values, index=formatted_postcodesectors)

            formatted_postcodesectors = [postcodesector.replace(" ", "")[:-1] + " " + postcodesector[-1] for postcodesector in formatted_postcodesectors]

            matching_postcodes = []

            for formatted_postcodesector in formatted_postcodesectors:
                
                query = f"SELECT postcode FROM self._postcodedb WHERE LEFT(postcode, CHAR_LENGTH(postcode) - 2) = '{formatted_postcodesector}'"
                result = pd.read_sql_query(query, self._postcodedb)

                matching_postcodes.extend(result['postcode'].tolist())

            # get median house price and store them in the series
            median_house_price_values = self.predict_median_house_price(matching_postcodes)
            median_house_price_series = pd.Series(data=median_house_price_values.values, index=formatted_postcodesectors)

            if len(households_series) != len(median_house_price_series):
                raise ValueError("Series lengths do not match.")

            return median_house_price_series * households_series


    def predict_annual_flood_risk(self, postcodes, risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(postcodes)
        total_properties=self.predict_total_value(postcodes)
        total_properties_aligned = total_properties.reindex(risk_labels.index)

        if len(risk_labels) != len(total_properties_aligned):
            raise ValueError("Series lengths do not match.")

        damage_coef=0.05

        return damage_coef*risk_labels*total_properties_aligned
