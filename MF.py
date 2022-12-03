import numpy as np

class MF(object):
    def __init__(self, Y, n_factors = 2, X = None, W = None, lamda = 0.1, learning_rate = 2, n_epochs = 50, top = 10):

        self.Y = Y
        self.lamda = lamda
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.top = top
        self.users_count = int(np.max(self.Y[:, 0])) + 1
        self.items_count = int(np.max(self.Y[:, 1])) + 1
        self.ratings_count = Y.shape[0]
        if X == None:
            self.X = np.random.randn(self.items_count, n_factors)
        if W == None:
            self.W = np.random.randn(n_factors, self.users_count)
        self.Ybar = self.Y.copy()
        self.bi = np.random.randn(self.items_count)
        self.bu = np.random.randn(self.users_count)
        self.n_ratings = self.Y.shape[0]

    def get_user_rated_item(self, i):
        ids = np.where(i == self.Ybar[:, 1])[0].astype(int)
        users = self.Ybar[ids, 0].astype(int)
        ratings = self.Ybar[ids, 2]
        return (users, ratings)

    def get_item_rated_by_user(self, u):
        ids = np.where(u == self.Ybar[:, 0])[0].astype(int)
        items = self.Ybar[ids, 1].astype(int)
        ratings = self.Ybar[ids, 2]
        return (items, ratings)

    def updateX(self):
        for m in range(self.items_count):
            users, ratings = self.get_user_rated_item(m)
            Wm = self.W[:, users]
            b = self.bu[users]
            sum_grad_xm = np.full(shape = (self.X[m].shape) , fill_value = 1e-8)
            sum_grad_bm = 1e-8
            for i in range(50):
                xm = self.X[m]
                error = xm.dot(Wm) + self.bi[m] + b - ratings
                grad_xm = error.dot(Wm.T)/self.n_ratings + self.lamda*xm
                grad_bm = np.sum(error)/self.n_ratings
                sum_grad_xm += grad_xm**2
                sum_grad_bm += grad_bm**2
                # gradient descent
                self.X[m] -= self.lr*grad_xm.reshape(-1)/np.sqrt(sum_grad_xm)
                self.bi[m] -= self.lr*grad_bm/np.sqrt(sum_grad_bm)
        
    def updateW(self):
        for n in range(self.users_count):
            items, ratings = self.get_item_rated_by_user(n)
            Xn = self.X[items, :]
            b = self.bi[items]
            sum_grad_wn = np.full(shape = (self.W[:, n].shape) , fill_value = 1e-8).T
            sum_grad_bn = 1e-8
            for i in range(50):
                wn = self.W[:, n]
                error = Xn.dot(wn) + self.bu[n] + b - ratings
                grad_wn = Xn.T.dot(error)/self.n_ratings + self.lamda*wn
                grad_bn = np.sum(error)/self.n_ratings
                sum_grad_wn += grad_wn**2
                sum_grad_bn += grad_bn**2
                # gradient descent
                self.W[:, n] -=round(self.lr*grad_wn.reshape(-1)/np.sqrt(sum_grad_wn),2)
                self.bu[n] -= round(self.lr*grad_bn/np.sqrt(sum_grad_bn),2)         

    def fit(self, x, data_size, Data_test, test_size = 0):
        for i in range(self.n_epochs):
            self.updateW()
            self.updateX()
            if (i + 1) % x == 0:
                self.RMSE(Data_test,data_size = data_size, test_size = 0, p = i+1)


    def pred(self, u, i):
        u = int(u)
        i = int(i)
        pred = self.X[i, :].dot(self.W[:, u]) + self.bi[i] + self.bu[u]
        
        return max(0, min(5, pred))
    
    def recommend(self, u):
        ids = np.where(self.Y[:, 0] == u)[0].astype(int)
        items_rated_by_user = self.Y[ids, 1].tolist()
        a = np.zeros((self.items_count,))
        recommended_items = []
        pred = self.X.dot(self.W[:, u])
        for i in range(self.items_count):
            if i not in items_rated_by_user:
                a[i] = pred[i] +self.bi[i] + self.bu[u]
        if len(a) < self.top:
            recommended_items = np.argsort(a)[-self.items_count:]
        else:
            recommended_items = np.argsort(a)[-self.top:]
        recommended_items = np.where(a[:] > 0)[0].astype(int)

        return recommended_items[:self.limit]



arr = np.array([
    [0,0,5],[0,1,4],[0,2,np.nan],[0,3,2],[0,4,2],
    [1,0,5],[1,1,np.nan],[1,2,4],[1,3,2],[1,4,0],
    [2,0,2],[2,1,np.nan],[2,2,1],[2,3,3],[2,4,4],
    [3,0,0],[3,1,0],[3,2,np.nan],[3,3,4],[3,4,np.nan],
    [4,0,1],[4,1,np.nan],[4,2,np.nan],[4,3,4],[4,4,np.nan],
    [5,0,np.nan],[5,1,2],[5,2,1],[5,3,np.nan],[5,4,np.nan],
    [6,0,np.nan],[6,1,np.nan],[6,2,1],[6,3,4],[6,4,5],
])

# mf = MF(arr)
# print(mf.W)
# print(mf.X)
















