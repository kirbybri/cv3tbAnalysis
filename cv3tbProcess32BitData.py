import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import pickle

#BEGIN CV3TB_PROCESS_32BITDATA CLASS
class CV3TB_PROCESS_32BITDATA(object):

  #__INIT__#
  def __init__(self, fileName = None):
    self.runResultsDict = None
    self.fileName = fileName
    self.chsIn32BitMode = []

    #calib constants
    self.chPairsIn32BitMode = {"channel1":("channel1","channel2"),
                              "channel2":("channel2","channel1"),
                              "channel3":("channel3","channel4"),
                              "channel4":("channel4","channel3"),
                              "channel5":("channel5","channel6"),
                              "channel6":("channel6","channel5"),
                              "channel7":("channel7","channel8"),
                              "channel8":("channel8","channel7"),
                              "SAR1":("channel1","channel2"),
                              "SAR8":("channel8","channel7"),
                              "DRE1":("channel1","channel2"),
                              "DRE2":("channel2","channel1"),
                              "DRE3":("channel3","channel4"),
                              "DRE4":("channel4","channel3"),
                              "MDAC1":("channel5","channel6"),
                              "MDAC2":("channel6","channel5"),
                              "MDAC3":("channel7","channel8"),
                              "MDAC4":("channel8","channel7"),
                              }

  def getAllChWfDict(self,measData):
    allChWfDict = {}
    for chDataDict in measData :
      if "channel" not in chDataDict or "wf" not in chDataDict:
        continue
      chId = chDataDict["channel"]
      if chId in allChWfDict:
        print("ERROR, repeated channel info")
        continue
      allChWfDict[chId] = chDataDict["wf"]
    return allChWfDict

  def convertIntTo16BitWord(self,sampInt):
    sampBin = bin(sampInt)[2:].zfill(16) #this is recovering the actual COLUTA data word which was stored as an int in the dict, this is silly
    return sampBin

  #assume only COLUTA, valid for CV3TB
  def getMeasCh32Bit(self,measData):
    allChWfDict = self.getAllChWfDict(measData)
    allChData32Bit = {}
    for ch in self.chsIn32BitMode:
      if ch not in self.chPairsIn32BitMode:
        print("ERROR, requested invalid ch for 32 bit mode")
        continue
      word0ChId = self.chPairsIn32BitMode[ch][0]
      word1ChId = self.chPairsIn32BitMode[ch][1]
      if word0ChId not in allChWfDict or word1ChId not in allChWfDict:
        #print("ERROR, does not have data required for 32 bit mode")
        continue
      word0ChData = allChWfDict[word0ChId]
      word1ChData = allChWfDict[word1ChId]
      if len(word0ChData) != len(word1ChData) :
        print("ERROR, mismatch in 32bit mode ch record lengths")
        continue
      chData32Bit = []
      for sampNum in range(0,len(word0ChData),1):
        samp0Bin = self.convertIntTo16BitWord(word0ChData[sampNum])
        samp1Bin = self.convertIntTo16BitWord(word1ChData[sampNum])
        samp32Bit = samp0Bin+samp1Bin
        chData32Bit.append(samp32Bit)
      if ch in allChData32Bit :
        print("ERROR, repeated data?")
        continue
      allChData32Bit[ch] = chData32Bit
    return allChData32Bit

  #output results dict to json file
  def outputFile(self):
    if self.runResultsDict == None:
      return None
    pathSplit = self.fileName.split('/')[-1]
    #jsonFileName = 'output_cv3tbProcessFile_' + pathSplit + '.json'
    #with open( jsonFileName , 'w') as outfile:
    #  json.dump( self.runResultsDict, outfile, indent=4)
    pickleFileName = 'output_cv3tbProcess32BitData_' + pathSplit + '.pickle'
    print("Output file ",pickleFileName)
    pickle.dump( self.runResultsDict, open( pickleFileName, "wb" ) )
    return

  def processFile(self):
    if self.runResultsDict == None:
      print("ERROR no input data")
      return None
    if "results" not in self.runResultsDict:
      print("ERROR invalid input data")
      return None
    runData = self.runResultsDict["results"]
    measResultsDict = {}
    for cnt, (measNum, measInfo) in enumerate(runData.items()):
      if "data" not in measInfo:
        continue
      measData = measInfo["data"]
      measAttrs = {}
      if "attrs" in measInfo:
        measAttrs = measInfo["attrs"]
      #print(cnt,"\t",measNum)
      allChData32Bit = self.getMeasCh32Bit(measData)
      measResultsDict[measNum] = {'data':allChData32Bit,'attrs':measAttrs}
    self.runResultsDict["results"] = measResultsDict
    return

  #open file
  def openFile(self):
    if self.fileName == None :
      print("ERROR no input file specified")
      return None
    self.runResultsDict = pickle.load( open( self.fileName, "rb" ) )
    return

def main():
  if len(sys.argv) != 2 :
    print("ERROR, program requires filename argument")
    return
  fileName = sys.argv[1]
  cv3tbProcess32Bit = CV3TB_PROCESS_32BITDATA(fileName)
  cv3tbProcess32Bit.openFile()
  #have to explicitly specify which channels are in 32 bit mode for now, get from metadata in future
  #cv3tbProcess32Bit.chsIn32BitMode = ["channel1","channel6","channel8"]
  cv3tbProcess32Bit.chsIn32BitMode = ["channel8"]
  cv3tbProcess32Bit.processFile()
  cv3tbProcess32Bit.outputFile()

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
