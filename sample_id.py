import os
import pandas as pd

##write sample files' names into a list

filePath = 'D:/PSU/DS 440/DataSet/ini_train'
resultList = []
for i,j,k in os.walk(filePath):
    fileList=k
fileList.pop()
for i in fileList:
    resultList.append(int(i.split('.')[0]))
print(resultList)

 
##write sample files name from list into csv file

# list.to_csv('e:/testcsv.csv',encoding='utf-8')
name=['train_id']
test=pd.DataFrame(columns=name,data=resultList)
print(test)
test.to_csv('D:/PSU/DS 440/DataSet/ini_trainFile_id.csv',encoding='gbk')
