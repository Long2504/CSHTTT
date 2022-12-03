import numpy as np

location = [['C1','C2','C1 C2'],['A1','A2','A3','A1 A2','A1 A3','A2 A3','A1 A2 A3']]

arr1 = [6,4,5]
arr2 = [[5,2,3,4,0,0,1],[3,1,2,3,3,1,2]]

tree = [location, [arr1, arr2]]
def dempster_shafer(tree, i):
    location = tree[0]
    arr1, arr2 = tree[1]

    location = np.array(location)
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    arr = arr2[i]
    tanxuat = arr/np.sum(arr1)
    bel = np.zeros(len(arr))
    pl = np.zeros(len(arr))
    for i in range(len(location[1])):
        node = location[1][i].split()
        index_node = set()
        index_node.add(i)
        index_giao_node = set()

        for j in range(len(location[1])):
            if set(location[1][j].split()).intersection(set(node)):
                index_giao_node.add(j)
            if set(location[1][j].split()).issubset(set(node)):
                index_node.add(j)
        index_giao_node = list(index_giao_node)
        index_node = list(index_node)
        bel[i] = np.sum(arr[index_node])/np.sum(arr1)
        pl[i] = np.sum(arr[index_giao_node])/np.sum(arr1)
    
    return tanxuat, bel, pl
p = [2/5,3/5]
arr = []
for i in range(len(arr2)):
    arr.append(dempster_shafer(tree,i))

bel_sm = np.zeros(len(arr2[0]))
pl_sm = np.zeros(len(arr2[0]))
for i in range(len(arr)):
    tanxuat, bel, pl = arr[i]
    bel_sm += p[i]*bel
    pl_sm += p[i]*pl

print(bel_sm,pl_sm)