import gensim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

from prepareData import w2X_train, w2X_test, w2y_train, w2y_test

vector_size = 130

w2v_model = gensim.models.Word2Vec(w2X_train,
                                   vector_size=vector_size,
                                   window=5,
                                   min_count=4)

print(w2v_model.wv.index_to_key)

words = set(w2v_model.wv.index_to_key)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in w2X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                        for ls in w2X_test])

# for i, v in enumerate(X_train_vect):
#     print(len(w2X_train.iloc[i]), len(v))

X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(vector_size, dtype=float))

X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(vector_size, dtype=float))

# for i, v in enumerate(X_train_vect_avg):
#     print(len(w2X_train.iloc[i]), len(v))

rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, w2y_train.values.ravel())

y_pred = rf_model.predict(X_test_vect_avg)

precision = precision_score(w2y_test, y_pred, average=None)
recall = recall_score(w2y_test, y_pred, average=None)

print('Precision: {} / Recall: {} / Accuracy: {}')
print(precision)
print(recall)

print((y_pred == w2y_test).sum()/len(y_pred))
