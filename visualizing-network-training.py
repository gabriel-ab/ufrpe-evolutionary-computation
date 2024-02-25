# %%
from matplotlib import pyplot as plt
from sklearn import datasets, model_selection, neural_network

# %%
X, y = datasets.make_blobs(3000, cluster_std=1.8, centers=5)
Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=.2)

# %% Display dataset
plt.scatter(*X.T, c=y, cmap='coolwarm')

# %%
model = neural_network.MLPClassifier().fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

# %%
plt.figure(figsize=(12, 5))
plt.subplot(121, title='Original').scatter(*Xtest.T, c=ytest, cmap='coolwarm')
plt.subplot(122, title='Predicted').scatter(*Xtest.T, c=ypred, cmap='coolwarm')
plt.show()
# %%
