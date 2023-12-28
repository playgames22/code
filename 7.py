import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
Y = pd.DataFrame(iris.target, columns=['Targets'])

print("Feature Data:")
print(X)
print("\nTarget Data:")
print(Y)

colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 3, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s=40)
plt.title('Real Clustering')

model1 = KMeans(n_clusters=3,n_init=10)
model1.fit(X)
plt.subplot(1, 3, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model1.labels_], s=40)
plt.title('K Means Clustering')

model2 = GaussianMixture(n_components=3)
model2.fit(X)
plt.subplot(1, 3, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model2.predict(X)], s=40)
plt.title('EM Clustering')

plt.show()

print("Actual Target is:\n", iris.target)
print("K Means:\n", model1.labels_)
print("EM:\n", model2.predict(X))
print("Accuracy of KMeans is ", sm.accuracy_score(Y, model1.labels_))
print("Accuracy of EM is ", sm.accuracy_score(Y, model2.predict(X)))
