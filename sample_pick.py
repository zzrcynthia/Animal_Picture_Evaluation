
import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir) 
        filenumber=len(pathDir)
        rate=0.4    #picking rate
        picknumber=int(filenumber*rate) #get the number of sample pictures
        sample = random.sample(pathDir, picknumber)  #randomly choose sample pictures based on pucknum
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = "D:/PSU/DS 440/DataSet/modified_petfinder-pawpularity-score/train/"    #initial folder path
	tarDir = 'D:/PSU/DS 440/DataSet/ini_train/'    #destination folder path
	moveFile(fileDir)















	
