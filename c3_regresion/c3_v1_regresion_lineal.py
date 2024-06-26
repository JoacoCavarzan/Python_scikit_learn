import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("random_data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data[["x"]],
    data["y"],
    test_size=0.3,
    random_state=42
)

regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(r2)

# print(
#     f="Coeficientes: {regression.coef_} \n"
#     f="Intercepto: {regression.intercept_}"
# )

plt.scatter(X_test, y_test, color="green")
plt.plot(X_test, y_pred, color="black")
plt.show()
