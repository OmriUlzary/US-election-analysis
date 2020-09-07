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


# In[19]:


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

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

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


# In[20]:


US_election_data_set_for_supervised


# In[28]:


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


# In[ ]:




