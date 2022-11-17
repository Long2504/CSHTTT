
import numpy as np

def AHP(arr):
    RI = [0, 0 , 0.52, 0.89, 1.11, 1.25, 1.35, 1.4, 1.45, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    sumCol = np.sum(arr,axis=0)
    arrNew = np.array([[i[j]/sumCol[j] for j in range(arr.__len__())] for i in arr])
    TS = np.array([[sum(i)/np.sum(arrNew)] for i in arrNew])
    TTS = np.dot(arr,TS)
    NQ = np.divide(TTS,TS)
    Lamda = np.average(NQ)
    CI = (Lamda - len(arr))/(len(arr)- 1)
    CR = CI/RI[len(arr)]
    if CR < 0.1 : return TS
    return 0

Location = ['A','B','C']

arr = np.array([
    [1, 1/5, 3, 4],
    [5, 1, 9, 7],
    [1/3, 1/9, 1, 2],
    [1/4, 1/7, 1/2, 1]
])
arrPrice = np.array([
    [1, 3, 2],
    [1/3, 1, 1/5],
    [1/2, 5, 1]
])
arrDistance = np.array([
    [1, 6, 1/3],
    [1/6, 1, 1/9],
    [3, 9, 1]
])
arrLabor = np.array([
    [1, 1/3, 1],
    [3, 1, 7],
    [1, 1/7, 1]
])
arrWage = np.array([
    [1, 1/3, 1/2],
    [3, 1, 4],
    [2, 1/4, 1]
])
arrAHP = AHP(arr)
arrPriceAHP = np.multiply(AHP(arrPrice),arrAHP[0])
arrDistanceAHP = np.multiply(AHP(arrDistance),arrAHP[1])
arrLaborAHP = np.multiply(AHP(arrLabor),arrAHP[2])
arrWageAHP = np.multiply(AHP(arrWage),arrAHP[3])


result = [arrPriceAHP[i] + arrDistanceAHP[i] + arrLaborAHP[i] + arrWageAHP[i] for i in range(len(arrDistanceAHP))]
print(Location[result.index(max(result))])