import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
'''
from sklearn.datasets import load_digits
digits = load_digits()
X= digits.data
y = digits.target
print(X.shape, y.shape)



'''

#import data
#contains 45,000 total rows of handwritten digits
#columns- total 785 {1(label)+ 784(28 x 28)--> each column denoting the pixel value}
data = pd.read_csv('train.csv')

X= data.iloc[:,1:].values
y = data.iloc[:,0].values

#normalize
#upgrading the pixel values for better clarity
X[X>90]=255 #adjusted for getting the maximum model accuracy

# plt.imshow(X[582].reshape((28,28)))
# plt.show()

#due to my system limitations I am using only 33% of the data for training and the rest is kept for testing.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.67)

classifier =SVC(kernel='rbf') #create the support vector machine model
classifier.fit(X_train,y_train) #train the model

# print(classifier.score(X_test[:100],y_test[:100]))
# for i in range(100):
# 	if classifier.predict([X_test[i]])[0] != y_test[i]:
# 		print("No match")
# 	else:
# 		print("match")	

print(accuracy_score(classifier.predict(X_test), y_test)) #accuracy

fname = 'my_model.pickle'
pickle.dump(classifier, open(fname,'wb')) #dump the model into a pickle file
