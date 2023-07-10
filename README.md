# Iris-flower-classification


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/Iris.csv")

df

df.head()
df = df.drop(columns = ['Id'])
df.head()

# to display stats about data
df.describe()


# to basic info about datatype  
df.info()



# to display no. of samples on each class 
df['Species'].value_counts()





# check for null values
df.isnull().sum()




#Histograms
df['SepalLengthCm'].hist()




df['SepalWidthCm'].hist()





df['PetalWidthCm'].hist()





df['PetalLengthCm'].hist()



#Scatterplot
colors = ['red', 'orange', 'blue']
Species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']





for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()



for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=Species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()



for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=Species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


#Coorelation Matrix
df.corr()




corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


#Label Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



df['Species'] = le.fit_transform(df['Species'])
df.head()



from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)



#Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()




#model training
model.fit(x_train, y_train)




#print metric to get performance
print("Accuracy: ",model.score(x_test, y_test)*100)





#knn - k-nearest neighbour
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()





model.fit(x_train, y_train)





#print metric to get performance
print("Accuracy:" ,model.score(x_test, y_test)*100)



model = DecisionTreeClassifier()  
model.fit(x_train, y_train)





#print metric to get performance
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy * 100)

