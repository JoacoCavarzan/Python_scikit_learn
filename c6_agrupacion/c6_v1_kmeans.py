"""    #No supervisado, agrupa lso datos en grupos/clusters basado en la similitud de caracteristicas

    1 Definir la cantidad de k de cluster
    2 inicializar de manera aleatoria los centros de cada cluster
    3 Asignar cada punto al centroide mas cercano
    4 Recalcular el centroide de cada clsuter
    5 Repetir 3 y 4 hasta que los centroides dejen de moverse o se alcance el limite de interaciones
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris_dataset = load_iris()
df_iris = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df_iris["target"] = iris_dataset.target

# inertia = []
# k_range = range(1, 11)
# for k in k_range:
#     kmeans = KMeans(random_state=5, max_iter=300, n_clusters=k)
#     kmeans.fit(iris_dataset.data)
#     inertia.append(kmeans.inertia_)

# plt.grid()
# plt.plot(range(1, 11), inertia, marker="o")
# plt.show()

kmeans = KMeans(random_state=5, max_iter=300, n_clusters=3)
kmeans.fit(iris_dataset.data)
print(kmeans.labels_)

df_iris["cluster"] = kmeans.labels_

grouped_iris = (
    df_iris
    .groupby(["target", "cluster"])
    .agg({"petal length (cm)": "count"})
)
print(grouped_iris)

