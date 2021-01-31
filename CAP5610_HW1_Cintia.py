# Machine Learning HW1 - Cintia Zuccon Buffon

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv(r'C:\Users\cinti\Documents\PythonF\DM\Titanic\train.csv')
test_df = pd.read_csv(r'C:\Users\cinti\Documents\PythonF\DM\Titanic\test.csv')
combine = [train_df, test_df]


## Question 5 - Check for blank, null or empty values
'''
# Training set
print(train_df.isnull().sum())

# Test set
print(test_df.isnull().sum())
'''


## Question 6 - data types
'''
print(train_df.dtypes)
'''


## Question 7 - Descriptive statistics - Numerical Features
'''
print(train_df[['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']].describe())
'''


## Question 8 - Descriptive statistics - Categorical Features
'''
# First transform the classes into categories 
train_df['Pclass'] = train_df['Pclass'].astype('category')
train_df['Name'] = train_df['Name'].astype('category')
train_df['Sex'] = train_df['Sex'].astype('category')
train_df['Ticket'] = train_df['Ticket'].astype('category')
train_df['Cabin'] = train_df['Cabin'].astype('category')
train_df['Embarked'] = train_df['Embarked'].astype('category')

# Verify
print(train_df.dtypes)

#Descriptive statistics - categorical
print(train_df[['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']].describe()) 
'''


## Question 9 - Correlation Pclass = 1 and Suvived
'''
# Count Plot 
sns.countplot(data = train_df , x = 'Pclass' , hue = 'Survived', palette = 'crest')
plt.show()

# Percentage Plot

sns.barplot(x='Pclass', y='Survived', data=train_df, palette = 'crest')
plt.show()
'''


## Question 10 - Correlation Sex = Female and Suvived
'''
# Count Plot
sns.countplot(data = train_df , x = 'Sex' , hue = 'Survived', palette = 'crest')
plt.show()

# Percentage Plot
sns.barplot(x='Sex', y='Survived', data=train_df, palette = 'crest')
plt.show()
'''


## Question 11 - Correlation Age x Survived
'''
# Plot 
grid = sns.FacetGrid(train_df, col='Survived')
grid.map(plt.hist, 'Age', alpha = 0.5)
grid.add_legend();
plt.show()

# Survival Rate Infants 
sum_infants = (train_df.loc[train_df['Age'] <= 4, 'Age'].count())
print(sum_infants)
sum_infants_survived= (train_df.loc[(train_df['Age'] <= 4) & (train_df['Survived']== 1),
    'Age'].count())
rate_survived_infants = sum_infants_survived / sum_infants
print(sum_infants_survived)
print(rate_survived_infants)

# Oldest Passenger is alive
print(train_df.loc[(train_df['Age'] == 80) & (train_df['Survived']== 1), 'Age'].count())

# Survival Rate 15-25 
sum_ages = (train_df.loc[(train_df['Age'] <= 25) & (train_df['Age'] >= 15), 'Age'].count())
print(sum_ages)
sum_ages_survived= (train_df.loc[(train_df['Age'] <= 25) & (train_df['Age'] >= 15) &
    (train_df['Survived']== 1), 'Age'].count())
rate_survived_age = sum_ages_survived / sum_ages
print(sum_ages_survived)
print(rate_survived_age)
'''


## Question 12 - Correlation Age x Survived x Class
'''
# Plot 
grid = sns.FacetGrid(train_df, col='Survived', row = 'Pclass')
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 10)
grid.add_legend();
plt.show()

# Class Rate
class_3_count = (train_df.loc[train_df['Pclass'] == 3, 'Pclass'].count())
class_3_survived = train_df.loc[(train_df['Pclass'] == 3) & (train_df['Survived']== 1),
    'Pclass'].count()
class_3_rate = class_3_survived / class_3_count
print(class_3_count)
print(class_3_rate)

# Infants survival class 2 and 3 
class_2_3_infants = train_df.loc[((train_df['Pclass'] == 3) & (train_df['Age'] <= 4))
    | (train_df['Pclass']== 2) & (train_df['Age'] <= 4),'Pclass'].count()
class_2_3_infants_survived = train_df.loc[((train_df['Pclass'] == 3) & (train_df['Age'] <= 4) &
    (train_df['Survived'] == 1)) | (train_df['Pclass']== 2) & (train_df['Age'] <= 4) &
    (train_df['Survived'] == 1),'Pclass'].count()
print(class_2_3_infants)
print(class_2_3_infants_survived)
'''


## Question 13 - Correlation Embarked x Sex x Fare and Survived 
'''
grid = sns.FacetGrid(train_df, col='Survived', row = 'Embarked')
grid.map(sns.barplot, 'Sex', 'Fare', order = ['female', 'male'], palette = 'crest')
grid.add_legend();
plt.show()
'''


# Question 14 - duplicates and correlation Ticket x Survival
'''
# Rate of duplicates 
duplicate = train_df.duplicated(subset=['Ticket'])
duplicates_ticket = duplicate.sum()
total_ticket = train_df['Ticket'].count()
rate = duplicates_ticket / total_ticket
print(total_ticket)
print(duplicates_ticket)
print(f'Rate of duplicates : {rate}' )

# Count Plot
sns.countplot(data = train_df , x = 'Ticket' , hue = 'Survived', palette = 'crest')
plt.show()

# Percentage Plot
sns.barplot(x='Ticket', y='Survived', data=train_df, palette = 'crest')
plt.show()
'''


# Question 15 - Cabine feature
'''
train_cabin = train_df['Cabin'].isnull().sum()
test_cabin = test_df['Cabin'].isnull().sum()
total_cabin = train_cabin + test_cabin
print(total_cabin)
'''

# Question 16 - Convert Sex to numerical values Male = 0 Female =1

'''
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
print(train_df['Sex'])
'''


## Question 17  - Complete missing values for Age feature
'''
# Used random numbers with mean and std
mean_age = train_df["Age"].mean()
std_age = train_df["Age"].std()
is_null_age = train_df["Age"].isnull().sum()
print(is_null_age)
random_age = np.random.randint(mean_age - std_age, mean_age + std_age, size = is_null_age)
age_slice = train_df["Age"].copy()
age_slice[np.isnan(age_slice)] = random_age
train_df["Age"] = age_slice
train_df["Age"] = train_df["Age"].astype(int)
is_null_age = train_df["Age"].isnull().sum()
print(is_null_age)
'''


## Question 18 - Most commom occurrences for Embarked

'''
most_commom = train_df['Embarked'].mode()
print(most_commom)
train_df['Embarked'] = train_df['Embarked'].fillna('S')
print(train_df['Embarked'].isnull().sum())
'''


## Question 19 - Complete Fare
'''
most_commom2 = test_df['Fare'].mode()
print(most_commom2)
test_df['Fare'] = test_df['Fare'].fillna(7.75)
print(test_df['Fare'].isnull().sum())
'''


##  Question 20 - Change values of Fare based on the table provided
'''
train_df.loc[train_df['Fare'] < 7.91, 'Fare'] = 0
train_df.loc[train_df['Fare'] < 14.454, 'Fare'] = 1
train_df.loc[train_df['Fare'] < 31, 'Fare'] = 2
train_df.loc[train_df['Fare'] >= 31, 'Fare'] = 3
'''

#train_df.to_csv('train_new.csv')
