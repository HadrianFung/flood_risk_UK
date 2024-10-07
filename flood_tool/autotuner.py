from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint as ran
from scipy.stats import uniform
from sklearn.linear_model import Lasso

class AutoTuner(object):
    def __init__(self, X, Y, pipeline,task='class'):
        '''
        Parameters
        ----------
        X : pandas DataFrame
            The data to be used for model selection
        Y : str
            The target column name
        pipeline : sklearn Pipeline
            The final pipeline used including preprocessing and the predictor model
        task : str, optional
            The task to be performed, by default 'class', or 'reg'
        '''
        self.X = X
        self.Y = Y
        self.pipeline = pipeline
        self.task = task
        # Define your models and parameters here
        self.models_parameters_classification = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=3000),
                'params': {
                    'C': uniform(0.1, 10),
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'class_weight': [None, 'balanced']
                }
            },
            'RandomForestClassifier' : {
            'model': RandomForestClassifier(),
                'params': {
                    'max_depth': ran(1,50),
                    'n_estimators': ran(100,500),
                    'min_samples_split': ran(2,10),
                    'max_features': ran(1,8),
                    'class_weight': [None, 'balanced', 'balanced_subsample']
                }
            },
            'KNeighborsClassifier': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': ran(1,10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            },
            'DecisionTreeClassifier': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': ran(1,50),
                    'min_samples_split': ran(2,10),
                    'max_features': ran(1,8),
                    'class_weight': [None, 'balanced']
                }
            },
            'SVC': {
                'model': SVC(probability=True),
                'params': {
                    'C': uniform(0.1, 10),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'class_weight': [None, 'balanced']
                }
            },
                'MLPClassifier': {
                'model': MLPClassifier(random_state=42, max_iter=200),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'early_stopping': [True],
                    'alpha': uniform(0.0001, 0.001),
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                }
            }
        }
        self.models_parameters_regression = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(),
                'params': {
                    'max_depth': ran(1, 50),
                    'n_estimators': ran(100, 500),
                    'min_samples_split': ran(2, 10),
                    'max_features': ran(1, 8)
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': uniform(0.1, 10),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': ran(1, 10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            },
            'XGBRegressor': {
                'model': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
                'params': {
                    'n_estimators': ran(100, 500),
                    'max_depth': ran(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.5, 0.5)
                }
            },
            'LassoRegressor': {
                'model': Lasso(),
                'params': {
                    'alpha': uniform(0.01, 10),
                    'fit_intercept': [True, False],
                    'max_iter': [1000, 2000, 3000],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'selection': ['cyclic', 'random']
                }
            }
        }

    def auto_tuning(self):
        '''
        Automatically tunes the model in the pipeline using RandomizedSearchCV.
        Returns
        -------
        sklearn Pipeline
            The final pipeline containing preprocessing and the tuned model with best hyper-parameters

        '''

        # Extract the model from the pipeline
        model_to_tune = self.pipeline.steps[-1][1]
        model_name = type(model_to_tune).__name__

        # Preprocess the data using the pipeline (excluding the last step)
        preprocessing_pipe = Pipeline(self.pipeline.steps[:-1])
        try:
            X_preprocessed = preprocessing_pipe.fit_transform(self.X)
            y_preprocessed = self.Y
        except TypeError:
            X_preprocessed = preprocessing_pipe.fit_transform(self.X, self.Y)
            y_preprocessed = self.Y
        # Select parameters dictionary
        if self.task == 'class':
            params_dict = self.models_parameters_classification
        elif self.task == 'reg':
            params_dict = self.models_parameters_regression

        # Match the model with its parameters
        if model_name not in params_dict:
            raise ValueError(f"Model {model_name} not found in parameters dictionary")
        model_params = params_dict[model_name]['params']

        # Perform RandomizedSearchCV
        search = RandomizedSearchCV(
            model_to_tune,
            param_distributions=model_params,
            n_iter=8,  # Reduced for less intensive computation
            cv=5,
            random_state=42,
            n_jobs=3  # Reduce parallel processing load
        )
        search.fit(X_preprocessed, y_preprocessed)

        # Return the best parameters and the tuned model
        best_params = search.best_params_
        print(f"Best parameters for {model_name}: {best_params}")
        self.tuned_model = model_name
        self.tuned_params = best_params
        # Update the model in the pipeline
        tuned_model = clone(model_to_tune).set_params(**best_params)
        self.pipeline.steps[-1] = (self.pipeline.steps[-1][0], tuned_model)
        temporary = self.pipeline.steps[:-1]
        from sklearn.pipeline import make_pipeline
        self.pipeline = make_pipeline(*temporary, tuned_model)
        return self.pipeline

    def get_tuned_model(self):
        '''
        Returns
        -------
        str
            The Model Name
        
        dictioanry
            The best hyper-parameters after tuning
        '''
        return self.tuned_model, self.tuned_params
