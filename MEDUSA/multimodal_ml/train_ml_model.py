from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from multimodal_ml.rf_opt import RF
from multimodal_ml.lr_opt import LR
import numpy as np
import pickle
from xgboost import XGBClassifier
from multimodal_ml.xgb_opt import XGB
import sys 
from sklearn.preprocessing import MinMaxScaler

classifier = sys.argv[1]
with open('multimodal_ml/embeddings/early_train.pkl', 'rb') as f:
    train_feat = pickle.load(f)

with open('multimodal_ml/embeddings/early_test.pkl', 'rb') as f:
    test_feat = pickle.load(f)

with open('../covid_log/covid_label_train.pkl', 'rb') as f:
    label_train = pickle.load(f)

with open('../covid_log/covid_label_test.pkl', 'rb') as f:
    label_test = pickle.load(f)

x_train = []
for i in train_feat:
      x_train.append(i[0])

x_test = []
for i in test_feat:
      x_test.append(i[0])

label2id = {}
id2label = {}
i = 0
for l in list(np.unique(label_train)):
        label2id[l] = i
        id2label[i] = l
        i = i + 1

label_train_int = []
for l in label_train:
        label_train_int.append(label2id[l])

label_test_int = []
for l in label_test:
        label_test_int.append(label2id[l])


scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#clf = LogisticRegression(random_state=42)
dist = (np.unique(label_test_int, return_counts=True))
class_weight = {0: ((len(x_train)) / (2 * dist[1][0])), 1: ((len(x_train)) / (2 * dist[1][1]))}
if classifier == 'rf':
    rf = RF(x_train, label_train_int)
    best = rf.find_best()
    clf = RandomForestClassifier(random_state=42, class_weight=class_weight, n_estimators=500, max_features=best['max_features'], n_jobs=-1)
elif classifier == 'lr':
    lr = LR(x_train, label_train_int)
    best = lr.find_best()
    clf = LogisticRegression(random_state=42, C = 2**best['C'], class_weight=class_weight)
elif classifier == 'xgb':
    xgb = XGB(x_train, label_train_int)
    best = xgb.find_best()
    y = np.array(label_train_int)
    clf = XGBClassifier(random_state = 42, scale_pos_weight=sum(y == 0) / sum(y == 1),
                                              objective='binary:logistic',
                                              n_estimators=500,
                                              learning_rate= best['learning_rate'],
                                              subsample=best['subsample'],
                                              max_depth=int(best['max_depth']),
                                              colsample_bytree=best['colsample_bytree'],
                                              min_child_weight=int(best['min_child_weight']), n_jobs=-1)
else:
    print('model not found')

clf.fit(x_train, label_train_int)
y_pred = clf.predict(x_test)

accuracy = classification_report(label_test_int, y_pred, digits=4)
print("CR:", accuracy)