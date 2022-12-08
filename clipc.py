import torch
import clip
from PIL import Image
import numpy as np
from composition_stats import alr

import os
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#################################################################################

filePath = 'D:/PSU/DS 440/DataSet/ini_train'        ##File ID
idList = []
for i,j,k in os.walk(filePath):
    fileList=k
fileList.pop()
for i in fileList:
    idList.append(int(i.split('.')[0]))
## print(idList)

idList.sort()
print(idList)
probList = []

###################################################################################

QUERIES = [                                         ##CLIP
    "sofa", 
    "bed", 
    "television", 
    "carpet", 
    "desk", 
    "tile", 
    "tree", 
    "swimming pool", 
    "grass", 
    "sky", 
    "collar", 
    "clothes", 
    "bandana", 
    "chain", 
    "leash", 
    "toy", 
    "men", 
    "women",
    "human",
    "single animal", 
    "multiple animals",
    "teeth", 
    "tongue",
    "cage",
    "bow", 
    "bowl", 
    "bone", 
    "treats",
    "yellow color",
    "red color",
    "white color",
    "black color",
    "greeen color",
    "blue color",
    "brown color",
    "pink color",
    "bush",
    "blanket",
    "cloud",
    "car",
    "floor",
    "bathroom",
    "garden",
    "soil",
    "flower",
    "grove",
    "towel",
    "pillow",
    "scarf",
    "hat",
    "pet clothing",
    "pet shoes",
    "pet food",
    "national flag",
    "potted plants",
    "cement floor",
    "plush toy",
    "child",
    "book",
    "sock",
    "box",
    "tablecloth",
    "wallpaper",
    "newspaper",
    "bookshelf",
    "sheet",
    "carton sticker",
    "stair",
    "bucket",
    "backpack",
    "leaf",
    "ball",
    "bell"
]


for i in idList:
    tempId = 'D:/PSU/DS 440/DataSet/ini_train/'+str(i)+'.jpg'               
    image = preprocess(Image.open(tempId)).unsqueeze(0).to(device)
    text = clip.tokenize(QUERIES).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probList.append(alr(probs))
##print(probList)
resultList = []
for i in range(len(probList)):
    temp = probList[i][0]
    resultList.append(temp)
print(resultList)

#####################################################################################
                                                     
name=[QUERIES]                                       ##Write into csv    
del(name[0])    
test=pd.DataFrame(columns=name,data=resultList)#esch column has a query as its name
test.to_csv('D:/PSU/DS 440/DataSet/ini_queryScore.csv',encoding='gbk')