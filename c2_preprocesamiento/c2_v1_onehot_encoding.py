import pandas as pd

from sklearn.preprocessing import OneHotEncoder

adult_df = pd.read_csv("adult.csv")
print(adult_df.columns)
print(adult_df.head())

print(adult_df.sex.unique())

one_hot_encoder = OneHotEncoder()
encode_adult_sex = one_hot_encoder.fit_transform(adult_df[["sex"]])
print(encode_adult_sex.toarray())
print(one_hot_encoder.categories_)

adult_df[one_hot_encoder.categories_[0]] = encode_adult_sex.toarray()
print(adult_df.columns)
