from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle

dataset = pd.read_csv('dataset/extracted_data.csv')
replace_rule = {'weather_condition' : {'rain': 1, 'sunshine' : 0}}
dataset = dataset.replace(replace_rule)

y = dataset['weather_condition']
x = dataset.iloc[:, :-1]

pca = PCA(n_components=2)
pca_x = pca.fit_transform(x)

# pca_df = {'pca_1': [], 'pca_2' : [], 'y': y}
pca_df = {'pca_1': [], 'pca_2' : []}

for item in pca_x :
    pca_df['pca_1'].append(item[0])
    pca_df['pca_2'].append(item[1])

pca_df = pd.DataFrame(pca_df)
# sns.relplot(data=pca_df, x='pca_1', y='pca_2', hue=y)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# Naive Bayes
start_time = time.time()
bnb_model = BernoulliNB()
bnb_model.fit(X_train, y_train)
print('BNB Elapsed ', time.time() - start_time)
y_pred_bnb = bnb_model.predict(X_test)

bnb_accuracy = accuracy_score(y_test, y_pred_bnb)
bnb_precision = precision_score(y_test, y_pred_bnb)
bnb_recall = recall_score(y_test, y_pred_bnb)
print("BNB Accuracy", bnb_accuracy, bnb_precision, bnb_recall)

# SVM
start_time = time.time()
svm_model = SVC()
svm_model.fit(X_train, y_train)
print('SVM Elapsed ', time.time() - start_time)
y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
print("SVM Accuracy", svm_accuracy, svm_precision, svm_recall)

# Decision Tree
start_time = time.time()
dst_model = DecisionTreeClassifier()
dst_model.fit(X_train, y_train)
print('DST Elapsed ', time.time() - start_time)
y_pred_dst = dst_model.predict(X_test)

dst_accuracy = accuracy_score(y_test, y_pred_dst)
dst_precision = precision_score(y_test, y_pred_dst)
dst_recall = recall_score(y_test, y_pred_dst)
print("DST Accuracy", dst_accuracy, dst_precision, dst_recall)

# Random Forest
rnd_model = RandomForestClassifier(max_depth=2)
rnd_model.fit(X_train, y_train)
y_pred_rnd = rnd_model.predict(X_test)

rnd_accuracy = accuracy_score(y_test, y_pred_rnd)
rnd_precision = precision_score(y_test, y_pred_rnd)
rnd_recall = recall_score(y_test, y_pred_rnd)
print("RND Accuracy", rnd_accuracy, rnd_precision, rnd_recall)

# Decision Tree (PCA)
X_train, X_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.2)
start_time = time.time()
dst_pca_model = DecisionTreeClassifier()
dst_pca_model.fit(X_train, y_train)
print('DST (PCA) Elapsed ', time.time() - start_time)
y_pred_pca_dst = dst_pca_model.predict(X_test)

dst_accuracy = accuracy_score(y_test, y_pred_pca_dst)
dst_precision = precision_score(y_test, y_pred_pca_dst)
dst_recall = recall_score(y_test, y_pred_pca_dst)
print("DST (PCA) Accuracy", dst_accuracy, dst_precision, dst_recall)

# Saving Model
model_file = open('models/dst.pickle', 'wb')
pickle.dump(dst_model, model_file)
model_file.close()

# Loading Model
model_file = open('models/dst.pickle', 'rb')
model = pickle.load(model_file)
model_file.close()