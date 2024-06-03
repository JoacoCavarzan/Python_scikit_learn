import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import normalize

array = np.array([1, 2, 3, 9, 6, 4, 5]) 
normalized_array = normalize([array])
print(normalized_array)

housing_dataset = fetch_california_housing(as_frame=True)
housing_df = housing_dataset.data
print(housing_df.head())

med_inc_array = np.array(housing_df["MedInc"])
normalized_med = normalize([med_inc_array])
print(normalized_med)

# housing_df["MedIncNorm"] = normalized_med[0]
# print(housing_df.head())

