import numpy as np
from scipy.sparse import coo_matrix
# print(np.zeros(6))

# a = np.array([ 5 , 4, np.nan,  2,  2])
# print(np.nansum(a)/5)


u0 = np.array([1.75, 0.75, np.nan,   -1.25, -1.25])
u1 = np.array([2.25, np.nan,    1.2, -0.75, -2.75])

test = np.where(np.isnan(u0),0,u0)

print(np.zeros((5,7)))

# def similarity(self):
#         self.S = self.dist_func(self.Ybar.T, self.Ybar.T)


# def distance( x1, x2):
#     return np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))

# print(distance(u0,u1))