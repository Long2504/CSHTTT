#from scipy.stats.stats import pearsonr
from scipy.sparse import coo_matrix
import numpy as np
from sklearn import metrics
import pandas as pd
# def pearson(X, Y = None):
#     x = X.shape[0]
#     y = X.shape[1]
#     a = np.zeros((x, x))
#     u = np.zeros((x, y))
#     temp = 0
    
#     for i in range(x):
#         for j in range(y):
#             u[i][j] = X[i, j]
#     for i in range(x):
#         for j in range(x):
#             temp = pearsonr(u[i], u[j])[0]
#             a[i][j] =  temp if not np.isnan(temp) else 0
    
#     return a


class NBCF:

    def __init__(self, Y, k, uuCF = 1, dist_f = metrics.pairwise.cosine_similarity, limit = 10):
        self.uuCF = uuCF
        self.f = open('danhgiaNBCF.dat', 'a+') 
        self.Y = Y if uuCF else Y[:, [1, 0, 2]]  #user_id,item_id,rating
        self.Ybar = None
        self.k = k
        self.limit = limit
        self.dist_func = dist_f
        self.users_count = int(np.max(self.Y[:, 0])) + 1
        self.items_count = int(np.max(self.Y[:, 1])) + 1
        self.Pu = None
        self.Ru = None


    def normalizeY(self):
            users = self.Y[:, 0]
            self.Ybar = self.Y.copy()
            self.mu = np.zeros((self.users_count,))
            for i in range(self.users_count):
                ids = np.where(users == i)[0].astype(int)
                ratings = self.Y[ids, 2]
                m = np.mean(ratings)
                if np.isnan(m):
                    m = 0
                self.mu[i] = m
                self.Ybar[ids, 2] = ratings - self.mu[i]
            self.Ybar = coo_matrix((self.Ybar[:, 2],
                (self.Ybar[:, 1], self.Ybar[:, 0])), (self.items_count, self.users_count))
            self.Ybar = self.Ybar.tocsr()

    def pred(self, u, i, normalized = 1):
        ids = np.where(self.Y[:, 1] == i)[0].astype(int)
        if ids == []:
            return 0
        users = (self.Y[ids, 0]).astype(int)
        sim = self.S[u, users]
        a = np.argsort(sim)[-self.k:]
        nearest = sim[a]
        r = self.Ybar[i, users[a]]
        
        if normalized:
            return (r*nearest)[0]/(np.abs(nearest).sum() + 1e-8)

        return (r*nearest)[0]/(np.abs(nearest).sum() + 1e-8) + self.mu[u]
        
        
    def _pred(self, u, i, normalized = 1):
        if self.uuCF: return self.pred(u, i, normalized)
        return self.pred(i, u, normalized)




# arr = np.array([
#     [0,0,5],[0,1,4],[0,2,None],[0,3,2],[0,4,2],
#     [1,0,5],[1,1,None],[1,2,4],[1,3,2],[1,4,0],
#     [2,0,2],[2,1,None],[2,2,1],[2,3,3],[2,4,4],
#     [3,0,0],[3,1,0],[3,2,None],[3,3,4],[3,4,None],
#     [4,0,1],[4,1,None],[4,2,None],[4,3,4],[4,4,None],
#     [5,0,None],[5,1,2],[5,2,1],[5,3,None],[5,4,None],
#     [6,0,None],[6,1,None],[6,2,1],[6,3,4],[6,4,5],
# ])
arr = np.array([
    [0,0,5],[0,1,4],[0,2,0],[0,3,2],[0,4,2],
    [1,0,5],[1,1,0],[1,2,4],[1,3,2],[1,4,0],
    [2,0,2],[2,1,0],[2,2,1],[2,3,3],[2,4,4],
    [3,0,0],[3,1,0],[3,2,0],[3,3,4],[3,4,0],
    [4,0,1],[4,1,0],[4,2,0],[4,3,4],[4,4,0],
    [5,0,0],[5,1,2],[5,2,1],[5,3,0],[5,4,0],
    [6,0,0],[6,1,0],[6,2,1],[6,3,4],[6,4,5],
])

#print(arr[:,0])

userBase = NBCF(arr,3,1)
userBase.normalizeY()
print(userBase.Ybar)

# test = [1 , 2 ,3 ,np.nan]
# print(sum(test))


