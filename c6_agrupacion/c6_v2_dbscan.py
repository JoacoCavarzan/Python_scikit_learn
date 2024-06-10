""" DBSCAN No Supervisado (Agrupacion espacial de aplicaciones con ruido basada en densidad)
    Identifica regiones densas en el dataset y las agrupa independiente
    -A diferencia del k-means: no requiere que se defina el nro de cluster
                               puede capturar cluster de forma comppleja
                               es capaz de indentificar outlayers(datos q no pertenecen a ningun cluster)

    Requiere 2 parametros:
    - eps-epsilon: indica el radio de cada cluster
    - minPts: nro de puntos min para considerar que es un punto core o nucleo

    DBSCAN Funciona:
    1-Seleccionar un dato de manera aleatoria y validar si es un punto nucleo
    2-Si es así, se crea un nuevo cluster y agregar los puntos alcanzables
    3-si el punto es borde, evaluar el siguiente punto
    4-el algoritmo se mueve por cda punto hasta evaluar todos los datos

"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=500, noise=0.1, random_state=10)
moons_df = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1]})

sns.scatterplot(data=moons_df, x="X1", y="X2")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=4)
y = dbscan.fit_predict(X_scaled)

moons_df["cluster"] = y

sns.scatterplot(data=moons_df, x="X1", y="X2", hue="cluster")
plt.show()


