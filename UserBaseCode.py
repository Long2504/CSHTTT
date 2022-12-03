
import numpy as np

class NBCF:

    def __init__(self, Y, k=2):
        self.Y = Y
        self.Ybar = None
        self.k = k
        self.users_count = int(np.max(self.Y[:, 0])) + 1
        self.items_count = int(np.max(self.Y[:, 1])) + 1
        self.arrnormalizeY = self.normalizeY()
        self.dist = self.similarity()

    def normalizeY(self):
        users = self.Y[:, 0]
        self.Ybar = self.Y.copy()
        arrNormalize = np.zeros((self.users_count,self.items_count))
        for i in range(self.users_count):
            ids = np.where(users == i)[0].astype(int)
            ratings = self.Y[ids, 2]
            #print(ratings)
            # self.Ybar[ids, 2] = np.round(ratings - np.nanmean(ratings),2)
            self.Ybar[ids, 2] = ratings - np.nanmean(ratings)
            arrNormalize[i,:] = np.where(np.isnan(self.Ybar[ids, 2]),0,self.Ybar[ids, 2])
        #print(self.mu)
        return arrNormalize

    def distance(self,x1, x2):
        return round(np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2)),2)

    def similarity(self):
        arrDist =  np.zeros((self.users_count,self.users_count))
        for i in range(self.users_count):
            for j in range(self.users_count):
                if(i == j):
                    arrDist[i][j] = 1
                else:
                    arrDist[i,j] = self.distance(self.arrnormalizeY[i],self.arrnormalizeY[j])
        return arrDist


    def pred(self, user, item):
        #get rating item for users
        arrRatingForItem = []
        for i in range(self.arrnormalizeY.__len__()):
            if self.arrnormalizeY[i][item] != 0:
                arrRatingForItem.append([self.arrnormalizeY[i][item],i])
        #print(arrRatingForItem,)

        sim = []
        def secondElement(elem):
            return elem[1]
        for i in range(arrRatingForItem.__len__()):
            i1 = self.dist[user, arrRatingForItem[i][1]]
            sim.append([arrRatingForItem[i][0], i1])

        sim.sort(key=secondElement,reverse=True)
        #print(sim)
        if self.k > sim.__len__(): self.k = sim.__len__()

        numerator = 0 
        denominator = 0
        for i in range(self.k):
            numerator = numerator + sim[i][1]*sim[i][0]
            denominator = denominator + abs(sim[i][1])
        return round(numerator/denominator,2)

    def recommend(self):
        recommended_items = np.zeros((self.items_count,self.users_count))
        for i in range(self.arrnormalizeY.__len__()):
            for j in range(self.arrnormalizeY[i].__len__()):
                if(self.arrnormalizeY[i][j] == 0):
                    recommended_items[j][i] = self.pred(i,j)
                else:
                    recommended_items[j][i] = round(self.arrnormalizeY[i][j],2)

        return recommended_items

arr = np.array([
    [0,0,5],[0,1,4],[0,2,np.nan],[0,3,2],[0,4,2],
    [1,0,5],[1,1,np.nan],[1,2,4],[1,3,2],[1,4,0],
    [2,0,2],[2,1,np.nan],[2,2,1],[2,3,3],[2,4,4],
    [3,0,0],[3,1,0],[3,2,np.nan],[3,3,4],[3,4,np.nan],
    [4,0,1],[4,1,np.nan],[4,2,np.nan],[4,3,4],[4,4,np.nan],
    [5,0,np.nan],[5,1,2],[5,2,1],[5,3,np.nan],[5,4,np.nan],
    [6,0,np.nan],[6,1,np.nan],[6,2,1],[6,3,4],[6,4,5],
])

userBase = NBCF(arr)
# userBase.normalizeY()
# userBase.similarity()
print(userBase.recommend())


