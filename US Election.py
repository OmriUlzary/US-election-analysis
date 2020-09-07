#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

def read_data(csv_file):
    try:
        return pd.read_csv(csv_file)
    except:
        print("The file is not found")
        return None


US_election_data_set = read_data("C:/Users/omri1/PycharmProjects/untitled2/primary_results.csv")


# In[2]:


US_election_data_set


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
import time 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import RFE

sns.jointplot(US_election_data_set['fraction_votes'], US_election_data_set['votes'], kind="regg", color="#ce1414")


# In[18]:


sns.set(style="white") 
sns.pairplot(US_election_data_set[['votes','fraction_votes']])


# In[ ]:





# In[4]:


from sklearn import preprocessing
import pandas as pd
import numpy as np

def pre_processing(original_data, supervised_or_unsupervised):
    def remove_columns(data_frame, col_name):
        try:
            columns = list(data_frame.columns)
            include_columns = [x for x in columns if x not in col_name]
            new_data_frame = data_frame[include_columns]
            return new_data_frame
        except:
            print('Something got wrong - remove_columns')

    name = original_data.isnull().sum().where(lambda x: x > 2500).dropna().keys().to_list()
    original_data = remove_columns(original_data, name)

    original_data = remove_columns(original_data, 'state_abbreviation')
    original_data = remove_columns(original_data, 'fips')

    if supervised_or_unsupervised == 'supervised':
        data_frame_for_supervised = original_data

        def encoders(data_frame):
            try:
                encoders = {
                    'party': preprocessing.LabelEncoder()
                }
                data_frame['party'] = encoders['party'].fit_transform(data_frame['party'].astype(str))
            except:
                print('Something got wrong - encoders')

        encoders(data_frame_for_supervised)
        
        def get_dummies(data_frame):
            try:
                data_frame = pd.get_dummies(data_frame)
                return data_frame
            except:
                print('Something got wrong - get_dummies')

        data_frame_for_supervised = get_dummies(data_frame_for_supervised)

        return data_frame_for_supervised

    if supervised_or_unsupervised == 'unsupervised':
        data_frame_for_unsupervised = original_data

        def get_dummies(data_frame):
            try:
                data_frame = pd.get_dummies(data_frame)
                return data_frame
            except:
                print('Something got wrong - get_dummies')

        data_frame_for_unsupervised = get_dummies(data_frame_for_unsupervised)

        return data_frame_for_unsupervised

    return original_data


US_election_data_set_for_supervised = pre_processing(US_election_data_set, 'supervised')

US_election_data_set_for_unsupervised = pre_processing(US_election_data_set, 'unsupervised')


# In[22]:


US_election_data_set_for_supervised


# In[6]:


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def naive_bayes_algorithm(data_frame):
    cols = list(data_frame.columns)
    cols.remove('party')

    X = data_frame[cols].copy()
    y = data_frame['party'].copy()

    def split_test_train(X, y, test_size):
        try:
            return train_test_split(X, y, test_size=test_size, random_state=0)
        except:
           print('Something got wrong - split_test_train')

    def create_naive_bayes_classifier(X, y):
        try:
            model = GaussianNB()
            model.fit(X, y)
            return model
        except:
            print('Something got wrong - create_naive_bayes_classifier')

    accuracy = []
    for ratio in np.arange(0.1, 0.5, 0.1):
        X_train, X_test, y_train, y_test = split_test_train(X, y, test_size = ratio)
        model = create_naive_bayes_classifier(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))

    best_accuracy = 0
    x = 0
    split = None
    for i in accuracy:
        x += 1
        if i > best_accuracy:
            best_accuracy = i
            split = "0." + str(x)
    print('The best accuracy is: ' + str(best_accuracy) +'\nThe size of the test team is: ' + str(split))

    ratios = np.arange(0.1, 0.5, 0.1)
    plt.grid(True)
    plt.plot(ratios, accuracy, 'r--')
    plt.xlabel('Size of Test Set')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Different Sizes of Train Set', fontsize=15)
    plt.show()

    X_train, X_test, y_train, y_test = split_test_train(X, y, test_size=0.3)
    model = create_naive_bayes_classifier(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.title('Confusion matrix for US election data frame')
    plt.show()


naive_bayes_algorithm(US_election_data_set_for_supervised)


# In[7]:


US_election_data_set_for_unsupervised


# In[8]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

def number_of_clusters(data_frame):
    sum_squared = []
    silhouette = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(data_frame)
        sum_squared.append(kmeans.inertia_)
        silhouette.append(silhouette_score(data_frame, kmeans.labels_))
    x1 = (range(2, 11))
    x2 = (range(2, 11))

    y1 = sum_squared
    y2 = silhouette

    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)
    plt.title(r'Sum of Squred Error ${R_2}$', fontsize = 15)
    # plt.xlabel('No. of Clusters')
    plt.ylabel(r'${R_2}$')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)
    plt.title('Silhouette',fontsize=15)
    plt.xlabel('No. of Clusters')
    plt.ylabel('Silhouette')

    plt.show()

def create_kmeans_calssifier(k):
    return KMeans(n_clusters=k, init='k-means++')

number_of_clusters(US_election_data_set_for_unsupervised)
kmeans = create_kmeans_calssifier(3)
kmeans.fit(US_election_data_set_for_unsupervised)
print(silhouette_score(US_election_data_set_for_unsupervised, kmeans.labels_))
US_election_data_set_for_unsupervised["label"] = pd.Series(kmeans.labels_)
clusters = US_election_data_set_for_unsupervised.groupby("label")
for name, group in clusters:
     print(name)
     print(group)
     print(US_election_data_set_for_unsupervised[US_election_data_set_for_unsupervised["label"] == name].describe())


# In[ ]:




