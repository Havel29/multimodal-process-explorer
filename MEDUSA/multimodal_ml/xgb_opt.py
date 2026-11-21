from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from hyperopt.pyll.base import scope
from sklearn.model_selection import StratifiedKFold


class XGB():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.space =  {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}


    def objective(self, params):

        self.y_train = np.array(self.y_train)
        scale_pos_weight = sum(self.y_train == 0) / sum(self.y_train == 1)
        model = XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate= params['learning_rate'],
                                                subsample=params['subsample'],
                                                max_depth=int(params['max_depth']),
                                                colsample_bytree=params['colsample_bytree'],
                                                min_child_weight=int(params['min_child_weight']),
                                                seed=42,scale_pos_weight=scale_pos_weight, n_jobs=-1
                                       )
        skf = StratifiedKFold(n_splits=3)

        accuracy = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='f1_macro').mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK}

    def find_best(self):
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials,rstate=np.random.default_rng(42))
        return best