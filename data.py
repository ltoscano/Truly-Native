import csv
import os.path
from zipfile import ZipFile

dataDir    = '.'

def setup(datadir="."):
    global dataDir
    dataDir = datadir;


def read_labels():
    global dataDir
    with open(os.path.join(dataDir, "train_v2.csv"), newline='') as f:
        f.readline() # skip header line
        labelReader = csv.reader(f)
        return dict((k, v == "1") for (k,v) in labelReader)
        
def read_samples(samplesName):
    #open files (let exceptions crash the whole process)
    files = [ZipFile(os.path.join(dataDir, str(i) + ".zip"), 'r') for i in range(0, 6)]
    samplesLoc = dict() # files => zip container pairs
    for i in range(0, 6):
        for f in files[i].namelist():
            samplesLoc[f[2:]] = i
    for s in samplesName:
        sampleLoc = samplesLoc[s]
        fileName = str(sampleLoc) + "/" + s
        with files[sampleLoc].open(fileName, 'r') as extractedFile:
            yield extractedFile.read().decode('UTF-8')
