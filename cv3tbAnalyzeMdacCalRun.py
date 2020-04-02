import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import glob
import pickle

class CV3TB_ANALYZE_MDACCAL(object):

  #__INIT__#
  def __init__(self, fileName = None):
    self.runResultsDict = None
    self.fileNameDict = None
    self.fileName = fileName
    self.dropInitialSamples = False
    self.numDroppedInitialSamples = 0
    self.applyCuts = False

    #calib constants
    self.mdacWeights = [0, 0, 0, 0, 0, 0, 0, 0]
    #self.sarWeights = [3584,2048,1024,640,384,256,128,224,128,64,32,24,16,10,6,4,2,1,0.5,0.25]
    self.sarWeights = [3584, 2048.65, 1025.31, 641.709, 385.31, 257.153, 128.205, 221.966, 127.491, 63.5096, 32.0504, 23.7532, 15.755, 9.8403, 5.87503, 3.89925, 1.9401, 1.00507, 0.516821, 0.30187]

  def getColutaSampleValue(self,sarBits=[],mdacBits=[]):
    val = 0
    for bitNum in range(len(sarBits)):
      val += self.sarWeights[bitNum]*int(sarBits[bitNum])
    for bitNum in range(len(mdacBits)):
      val += self.mdacWeights[bitNum]*int(mdacBits[bitNum])
    return val

  def getWaveformValsFrom32BitData(self,colutaWf):
    vals = []
    for samp in colutaWf :
      header = samp[0:2]
      clockCheck = samp[2:4]
      mdacBits = samp[4:12]
      sarBits = samp[12:32]
      sarBitNum = int(sarBits,2)
      val = self.getColutaSampleValue(sarBits,mdacBits)
      vals.append(val)
    return vals

  def plotVals(self,measNum,vals,psd_x,psd):
    #return
    fig, axes = plt.subplots(1,2,figsize=(10, 6))
    axes[0].plot(vals,".")
    axes[0].set_xlabel('Sample #', horizontalalignment='right', x=1.0)
    axes[0].set_ylabel('ADC CODE [ADC]', horizontalalignment='left', x=1.0)
    axes[0].set_title("COLUTA WAVEFORM, " + str(measNum) )
    #axes[0].set_xlim(0,1000)
    axes[1].plot(psd_x,psd,".")
    axes[1].set_xlabel('Frequency [MHz]', horizontalalignment='right', x=1.0)
    axes[1].set_ylabel('PSD [dB]', horizontalalignment='left', x=1.0)
    axes[1].set_title("")
    fig.tight_layout()
    plt.show()

  def getMeasChData(self,chId=None,measNum=None):
    if chId == None or self.runResultsDict == None or measNum == None:
      return None
    if "results" not in self.runResultsDict:
      return None
    runData = self.runResultsDict["results"]
    if measNum not in runData :
      return None
    measInfo = runData[measNum]
    if "data" not in measInfo:
      return None
    measData = measInfo["data"]
    if chId not in measData:
      return None
    #if "attrs" in measInfo:
    #  print(measNum,"\t", measInfo["attrs"] )

    chWf = measData[chId]
    #account for bad samples
    if self.dropInitialSamples :
      chWf = chWf[self.numDroppedInitialSamples:]
    return chWf

  def processMdacCalRun(self,chId=None):
    if chId == None or self.runResultsDict == None :
      return None
    if "results" not in self.runResultsDict:
      return None
    runData = self.runResultsDict["results"]
    reqMeas = ["Measurement_0","Measurement_1","Measurement_2","Measurement_3","Measurement_4",
               "Measurement_5","Measurement_6","Measurement_7","Measurement_8","Measurement_9",
               "Measurement_10","Measurement_11","Measurement_12","Measurement_13","Measurement_14","Measurement_15"]
    #check if all required measurements are recorded
    meanVals = {}
    rmsVals = {}
    for measNum in reqMeas :
      if measNum not in runData :
        return None
      measInfo = runData[measNum]
      if "data" not in measInfo:
        return None
      measData = measInfo["data"]
      if chId not in measData:
        return None
      chWf = self.getMeasChData(chId,measNum)
      if chWf == None :
        return None
      vals = self.getWaveformValsFrom32BitData(chWf)
      meanVals[measNum] = np.mean(vals)
      rmsVals[measNum] = np.std(vals)
      #self.viewWaveform(chId,measNum)
    
    #check if all required measurements in results
    for measNum in reqMeas :
      if measNum not in meanVals :
        return None
  
    for measNum in reqMeas :
      print(measNum,"\t",meanVals[measNum],"\t",rmsVals[measNum])

    #calculate MDAC constants
    mdacVals = []
    '''
    mdacVals.append( meanVals['Measurement_1'] -  meanVals['Measurement_0'] )
    mdacVals.append( meanVals['Measurement_3'] -  meanVals['Measurement_2'] )
    mdacVals.append( meanVals['Measurement_5'] -  meanVals['Measurement_4'] )
    mdacVals.append( meanVals['Measurement_7'] -  meanVals['Measurement_6'] )
    mdacVals.append( meanVals['Measurement_9'] -  meanVals['Measurement_8'] )
    mdacVals.append( meanVals['Measurement_11'] -  meanVals['Measurement_10'] )
    mdacVals.append( meanVals['Measurement_13'] -  meanVals['Measurement_12'] )
    mdacVals.append( meanVals['Measurement_15'] -  meanVals['Measurement_14'] )
    '''
    mdacVals.append( meanVals['Measurement_0'] -  meanVals['Measurement_1'] )
    mdacVals.append( meanVals['Measurement_2'] -  meanVals['Measurement_3'] )
    mdacVals.append( meanVals['Measurement_4'] -  meanVals['Measurement_5'] )
    mdacVals.append( meanVals['Measurement_6'] -  meanVals['Measurement_7'] )
    mdacVals.append( meanVals['Measurement_8'] -  meanVals['Measurement_9'] )
    mdacVals.append( meanVals['Measurement_10'] -  meanVals['Measurement_11'] )
    mdacVals.append( meanVals['Measurement_12'] -  meanVals['Measurement_13'] )
    mdacVals.append( meanVals['Measurement_14'] -  meanVals['Measurement_15'] )
    print("MDAC VALS",mdacVals)
    return

  def viewWaveform(self,chId=None,measNum=None):
    chWf = self.getMeasChData(chId,measNum)
    if chWf == None :
      return None
    vals = self.getWaveformValsFrom32BitData(chWf)
    #print("Number of samples ",len(chWf))
    print("MEAS#\t",measNum,"\tMEAN\t",np.mean(vals),"\tRMS\t",np.std(vals) )
    psd_x = []
    psd = []
    self.plotVals(measNum,vals,psd_x,psd)

  def dumpFile(self):
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    for measNum in runData:
      measData = runData[measNum]
      print("Measurement ", measNum)
      print("\t",measData)
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
  cv3tbAnalyzeFile = CV3TB_ANALYZE_MDACCAL(fileName)
  #cv3tbAnalyzeFile.openFile()
  #cv3tbAnalyzeFile.dumpFile()
  #cv3tbAnalyzeFile.printMeasSamples("channel8","Measurement_1")
  #cv3tbAnalyzeFile.viewWaveform("channel5","Measurement_14")
  #for measNum in cv3tbAnalyzeFile.runResultsDict["results"] :
  #  cv3tbAnalyzeFile.viewWaveform("channel8",measNum)
  #  #cv3tbAnalyzeFile.printEnob("channel5",measNum)
  #for measNum in cv3tbAnalyzeFile.runResultsDict["results"] :
  #  cv3tbAnalyzeFile.printMeasSamples("channel8",measNum)

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
