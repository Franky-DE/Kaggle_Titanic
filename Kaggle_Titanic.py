
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv as csv

df_train_set = pd.read_csv('train.csv')

df_train_set


# In[ ]:

df_train_set.describe()


# In[ ]:

df_train_set.dtypes


# In[ ]:

df_train_set['Sex_bin']=df_train_set['Sex']
df_train_set.replace({'Sex_bin' : { 'male' : 1 }}, inplace=True)
df_train_set.replace({'Sex_bin' : { 'female' : 0 }}, inplace=True)
df_train_set


# In[ ]:

df_train_set.describe()


# In[ ]:

df_train_set_male = df_train_set[df_train_set['Sex_bin'] == 1]
df_train_set_male.describe()


# In[ ]:

df_train_set_female = df_train_set[df_train_set['Sex_bin'] == 0]
df_train_set_female.describe()


# In[ ]:

df_train_set_male_cl1 = df_train_set_male[df_train_set['Pclass'] == 1]
df_train_set_male_cl1.describe()


# In[ ]:

df_train_set_male_cl2 = df_train_set_male[df_train_set['Pclass'] == 2]
df_train_set_male_cl2.describe()


# In[ ]:

df_train_set_male_cl3 = df_train_set_male[df_train_set['Pclass'] == 3]
df_train_set_male_cl3.describe()


# In[ ]:

df_train_set_female_cl1 = df_train_set_female[df_train_set['Pclass'] == 1]
df_train_set_female_cl1.describe()


# In[ ]:

df_train_set_female_cl2 = df_train_set_female[df_train_set['Pclass'] == 2]
df_train_set_female_cl2.describe()


# In[ ]:

df_train_set_female_cl3 = df_train_set_female[df_train_set['Pclass'] == 3]
df_train_set_female_cl3.describe()


# In[ ]:

def func_pred1(Pclass, Sex, RandNumb):
    if Sex == 'female':
        if Pclass == 1 and RandNumb < 0.968085:
            prob = 1
        elif Pclass == 2 and RandNumb < 0.921053:
            prob = 1
        elif Pclass == 3 and RandNumb < 0.5:
            prob = 1
        else:
            prob = 0
    else:
        if Pclass == 1 and RandNumb < 0.368852:
            prob = 1
        elif Pclass == 2 and RandNumb < 0.157407:
            prob = 1
        elif Pclass == 3 and RandNumb < 0.135447:
            prob = 1
        else:
            prob = 0
    return prob


# In[ ]:

from pandas import Series

test_file = open('test.csv', 'rb')
read_lines_test = test_file.readlines()
count_lines_test = len(read_lines_test)-1
test_file.close()
np.random.seed(1)

Ser_rand_test = Series(np.random.rand(count_lines_test))

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

predictions_file = open("genderclassmodel_fb.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])


for row in test_file_object:

    predictions_file_object.writerow([row[0], func_pred1(int(row[1]),row[3], Ser_rand_test[test_file_object.line_num - 2])])

test_file.close()
predictions_file.close()    


# In[ ]:




# In[ ]:




# In[ ]:



