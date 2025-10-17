#NUMERICAL DATA:
#1.NORMALIZATION
#2.STANDARDIZATION
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

data = np.array([[10],[20],[30],[40],[500]])

norm = MinMaxScaler()
print("Normalization:",norm.fit_transform(data))

std = StandardScaler()
print("Standardized:",std.fit_transform(data))

#CATEGORICAL DATA:
#ONE HOT ENCODING

from sklearn.preprocessing import OneHotEncoder
import numpy as np

colors= np.array([['blue'],['orange'],['pink'],['white']])
encoder = OneHotEncoder(sparse_output = False)
encoded = encoder.fit_transform(colors)
print(encoded)

#TEXTUAL DATA:
#TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

docs = ['I love cats','I love dogs','Dogs love me']

vectorizor = TfidfVectorizer()
X=vectorizor.fit_transform(docs)

print(vectorizor.get_feature_names_out())
print(X.toarray())