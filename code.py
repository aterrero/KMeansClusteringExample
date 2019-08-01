import operator
import pandas as pd
import numpy as np
from scipy.io import arff

#Loading the data
maindata = arff.loadarff("segment.arff")
maindf = pd.DataFrame(maindata[0])
maindf['class'] = maindf['class'].str.decode("utf-8")
maindf['centroid'] = ['']*len(maindf)
columnlist = list(maindf.columns)
columnlist.remove('class')
columnlist.remove("centroid")
normalizeddf = maindf.copy(deep = True)

#Normalization of the data
for i in range(0,19):
    values = np.matrix(maindf[columnlist[i]])
    values = values.T
    average = values.mean()
    stdeviation = values.std()
    values2 = (values-average)/stdeviation
    normalizeddf[columnlist[i]] = values2

normalizeddf['region-pixel-count'] = [0]*len(normalizeddf)
normalizeddf['centroid'] = ['']*len(normalizeddf)

#Startingpositions list
startingpositions = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 
                     1160, 108, 422, 88, 105, 261, 212,1941, 1724, 704, 1469,
                     635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812,
                     214,53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711,
                     1997, 1378, 827, 1875, 424,1790, 633, 208, 1670, 1517, 1902, 
                     1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 672, 483,65, 92,
                     400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954,
                     1818, 1679, 832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640,
                     833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 520, 1363,
                     544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 
                     1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203,
                     1068, 1493, 859, 233, 1846, 1119, 469, 1869, 609, 385, 1182, 
                     1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339,
                     826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692,
                     178, 774, 1623, 1641, 661, 1242, 1757, 553, 1377, 1419, 306, 
                     1838, 211, 356, 541, 1455, 741, 583, 1464, 209, 1615, 475,
                     1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785,
                     298, 1485, 945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690,
                     1277, 345, 1166, 1367, 1858, 1427, 1434, 953, 1992, 1140,
                     137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240,
                     524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561,
                     1270, 1250, 613, 152, 1440, 473, 1834, 1387, 1656, 1028, 1106, 829,
                     1591, 1699, 1674, 947, 77, 468, 997, 611, 1776, 123, 979, 1471, 1300,
                     1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686, 854, 257,
                     1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582, 816, 1770, 663,
                     737, 1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 1472]


# To store SSEs, I will create a dictionary for each k, the value is going to be a list
#    and I will append the SSE of each trial, for each K        
SSEdict = {}
for i in range(12):
    SSEdict["k={0}".format(i+1)] = []
    
for x in range(1,13):
    k = x

    indexlistlength = len(startingpositions)
    initialcentroidlist = []
    for i in range(0,25):
        print (startingpositions[i*k:(i*k)+k])
        initialcentroidlist.append(startingpositions[i*k:(i*k)+k])

    for p in range(len(initialcentroidlist)):  
    #   Creating the centroids
        centroiddict = {}
        for i in range(k):
            centroiddict["centroid{0}".format(i+1)] = np.matrix(normalizeddf[columnlist].loc[initialcentroidlist[p][i]])
    
    #   Preparing environment for k means clustering
        datamatrix = np.matrix(normalizeddf[columnlist])
        oldpredictions = ['c0' for i in range(len(normalizeddf))]
        newpredictions = ['c1' for i in range(len(normalizeddf))]
        counter = 0

    #   Implementation of the k means algorithm
        while oldpredictions != newpredictions:
            counter = counter + 1
            print("This is iteration", counter, "of trial",p+1,"with k =",k)
            oldpredictions = newpredictions
    
    #       This for loop is for computing the distances and generating the predictions
            for i in range(len(datamatrix)):
                distancedict = {}
                for j in range(k):
                    distancedict["distc{0}".format(j+1)] = np.sum(np.square(datamatrix[i]-centroiddict["centroid{0}".format(j+1)]))
                distancelist = sorted(distancedict.items(), key = operator.itemgetter(1))
                newcentroid = distancelist[0][0][4:]
                normalizeddf.at[i,'centroid'] = newcentroid
            newpredictions = list(normalizeddf['centroid'].values)
    
    #       To recalculate the centroids
            centroidinstances = {}
            for i in range(k):
                centroidinstances["c{0}instances".format(i+1)] = normalizeddf['centroid'] == "c{0}".format(i+1)
                centroiddict["centroid{0}".format(i+1)] = np.matrix(normalizeddf[centroidinstances["c{0}instances".format(i+1)]][columnlist]).mean(axis = 0)                       
            
    #       To break in case the algorithm reaches 50 iterations
            if counter == 50:
                break 
        
    #   To calculate SSE
        SSElist = []
        for i in range(k):
            centroidSSE = np.sum(np.square(np.matrix(normalizeddf[columnlist][centroidinstances["c{0}instances".format(i+1)]]) - centroiddict["centroid{0}".format(i+1)]))
            SSElist.append(centroidSSE)
        trialSSE = sum(SSElist)
        SSEdict["k={0}".format(k)].append(trialSSE)
    

resultdf = pd.DataFrame(['0']*12)
resultdf.columns = ['k']
resultdf['MuK - 2sigma'] = ['0']*12
resultdf['MuK'] = ['0']*12
resultdf['MuK + 2sigma'] = ['0']*12
for i in range(len(resultdf)):
    mean = np.mean(SSEdict['k={0}'.format(i+1)])
    stdeviation = np.std(SSEdict['k={0}'.format(i+1)])
    resultdf.at[i,'k'] = i+1
    resultdf.at[i,'MuK'] = mean
    resultdf.at[i,'MuK - 2sigma'] = mean - 2*stdeviation
    resultdf.at[i,'MuK + 2sigma'] = mean + 2*stdeviation
resultdf = resultdf.set_index('k')
resultdf.plot()
print(resultdf)
