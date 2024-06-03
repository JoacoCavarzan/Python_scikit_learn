import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()
iris_data = iris_dataset.data

fig, ax = plt.subplots()
scatter = ax.scatter(iris_data[:, 0], iris_data[:, 1], c=iris_dataset.target)
plt.show()

iris_df = pd.DataFrame(iris_data, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_df, c=iris_dataset.target)
plt.show()