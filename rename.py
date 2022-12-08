import os

path = 'D:/PSU/DS 440/DataSet/modified_petfinder-pawpularity-score/train'
i = 0
picList = sorted(os.listdir(path))
print(picList)

for pic in picList:
    os.rename(os.path.join(path, pic), os.path.join(path, (str(i)+'.jpg')))
    i += 1
