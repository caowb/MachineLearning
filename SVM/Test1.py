from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np

x = np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y = np.array([1,1,2,2])

# clf = SVC()
clf = NuSVC()
# clf.fit(x,y)
print clf.fit(x,y)
print clf.predict([[-0.8,-1]])
print clf.predict([[-0.8,-12]])
print clf.predict([[-0.18,-1]])
