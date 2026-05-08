import pandas as pd # these are the same as Lab 6

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

hd = pd.read_csv('data/heart.csv')

#Changing column names 

hd.columns = [
    'age', 
    'sex', 
    'chest_pain_experienced', 
    'resting_blood_pressure', 
    'cholesterol', 
    'fasting_blood_sugar', 
    'electrocardiograph',
    'maximum_heart_rate',
    'exercise_induced_angina',
    'st_depression',
    'st_slope',
    'major_vessels',
    'thalassemia',
    'target'
]

#choosing only men's data
men = hd[hd['sex']==1]

#Printing average data of men's variable who experienced chest pain
means_men= men.groupby(by='chest_pain_experienced').mean()
#print(means_men)

women = hd[hd['sex']== 0]
means_women = women.groupby(by='chest_pain_experienced').mean()
#print(means_women)

sns.countplot(x='target', data=hd)

sns.scatterplot(data=hd[hd['age'].between(10,70)], x='maximum_heart_rate', y='age', hue='target')


from scipy import stats

#print(stats.pearsonr(hd['maximum_heart_rate'], hd['age']))


#Classifier
from sklearn.ensemble import RandomForestClassifier # import sklearn
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100, random_state=0) 


hrd=hd.sample(frac=1)
train= hrd[:240]
test= hrd[240:303]
print('Train: ' + str(len(train)) + ' Test: ' + str(len(test)) )

training_observed= train.drop('target', axis=1)
training_labels=train['target']

clf.fit(training_observed, training_labels)


test_observed = test.drop('target', axis=1)
pred= clf.predict(test_observed)

print(pred)

test_labels=test['target']
accuracy_score=accuracy_score(test_labels, pred)
print(accuracy_score)




# age: The person's age in years
# sex: The person's sex (1 = male, 0 = female)
# cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# chol: The person's cholesterol measurement in mg/dl
# fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# thalach: Maximum heart rate achieved
# exang: Exercise induced angina (1 = yes; 0 = no)
# oldpeak: ST depression induced by exercise relative to rest
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# ca: The number of major vessels (0-3)
# thal: A blood disorder called thalassemia (0 = fixed defect; 1 = normal defect; 2 = reversable defect __Note:__ changed for our class for simplicity)
# target: Heart disease (0 = no, 1 = yes)