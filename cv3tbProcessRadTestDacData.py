import json
import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt

from cv3tbProcessFile import CV3TB_PROCESS_FILE
from cv3tbProcess32BitData import CV3TB_PROCESS_32BITDATA
from cv3tbAnalyzeSineWave import CV3TB_ANALYZE_SINEWAVE
from cv3tbAnalyzeWaveform import CV3TB_ANALYZE_WAVEFORM

#BEGIN CLASS
class CV3TB_PROCESS_RADTESTDAC(object):
  #__INIT__#
  def __init__(self, fileName = None):
    #global objects/variables
    self.fileName = fileName
    self.cv3tbProcessFile = CV3TB_PROCESS_FILE(self.fileName)
    self.cv3tbProcess32Bit = CV3TB_PROCESS_32BITDATA(self.fileName)
    #self.cv3tbAnalyzeFile = CV3TB_ANALYZE_SINEWAVE(self.fileName)
    self.cv3tbAnalyzeFile = CV3TB_ANALYZE_WAVEFORM(self.fileName)
    self.runResultsDict = None
    self.numSampleReq = 4096
    self.numSampleSkip = 0
    self.reqLengthSysComments = 6

    self.recMeasNum = []
    self.recChName = []
    self.recItr = []
    self.recAmp = []
    self.recEnob = []
    self.recTimestamp = []
    self.measResults = {}
    self.minTimestamp = None

    #print some info about what the program expects
    #print("PROCESSING RAD TEST DATA")
    #print("\tEXPECT ",self.numSampleReq," SAMPLES IN EACH READOUT")    

  #extract required info from sysComments metadata
  def parseSysComments(self,measNum=None,sysComments=None):
    if measNum == None or sysComments == None:
      print("ERROR parseSysComments, invalid input",measNum,sysComments)
      return None
    sysComments = sysComments.split(";")
    #if len(sysComments) != self.reqLengthSysComments:
    if len(sysComments) < 3 :
      print("ERROR parseSysComments, metadata field sysComments does not have expected length,",measNum)
      return None
    return sysComments[2]

  def getMeasData(self,measNum=None,measInfo=None):
    if measNum == None or measInfo == None:
      print("ERROR getMeasData, invalid input")
      return None
    if "data" not in measInfo:
      print("ERROR getMeasData, measurement does not contain data,",measNum)
      return None
    measData = measInfo["data"]
    if "attrs" not in measInfo:
      print("ERROR getMeasData, measurement does not contain metadata,",measNum)
      return None
    measAttrs = measInfo["attrs"]
    if "sysComments" not in measAttrs:
      print("ERROR getMeasData, required field sysComments missing from metadata,",measNum)
      return None
    sysComments = measAttrs["sysComments"]
    #return measData,sysComments
    return measData,measAttrs

  def dumpData(self,measNum=None,measData=None):
    if measNum == None or measData == None:
      return None
    for meas in measData:
      if ("coluta" not in meas) or ("channel" not in meas) or ("wf" not in meas) :
        print("WEIRD",meas)
        continue
      print(meas["channel"])
      chData = meas["wf"]
      chWf = []
      for sampNum in range(0,len(chData),1):
        samp = self.cv3tbProcess32Bit.convertIntTo16BitWord(chData[sampNum])
        chWf.append(samp)
      print("\t",chWf[0:10]) #raw ch 16-bit data word
      continue
      vals = self.cv3tbAnalyzeFile.getWaveformValsFrom32BitData(chWf)
      psd_x,psd,sinad,enob = self.cv3tbAnalyzeFile.getFftWaveform(vals)
      self.cv3tbAnalyzeFile.plotVals(measNum,vals,psd_x,psd)
    return None

  def printMeasResults(self):
    orderMeasNum = np.argsort(self.recMeasNum)
    recMeasNumOrd = np.array(self.recMeasNum)[orderMeasNum]
    recChNameOrd = np.array(self.recChName)[orderMeasNum]
    recItrOrd = np.array(self.recItr)[orderMeasNum]
    recAmpOrd = np.array(self.recAmp)[orderMeasNum]
    recEnobOrd = np.array(self.recEnob)[orderMeasNum]
    for num in range(0,len(recMeasNumOrd),1):
      print("Measurement:\t",recMeasNumOrd[num],"\tCh:",recChNameOrd[num],"\tIter:",recItrOrd[num],"\tAmp:",recAmpOrd[num],"\tENOB:",recEnobOrd[num])
    return

  def plotMeasResults(self):
    #dimensions of summary plots 
    numRows = 2
    numCols = 4

    #dict of required results, organized by plot panel
    chList = ["channel1","channel2","channel3","channel4","channel5","channel6","channel7","channel8"]
    #chList = ["channel6"]
    chListData = {}
    for ch in chList:
      if ch not in self.measResults :
        chListData[ch] = {}
        continue  
      chListData[ch] = {'ch':ch,'dacList':[dac for dac in self.measResults[ch] ],'meanList':[self.measResults[ch][dac]["mean"] for dac in  self.measResults[ch] ]}  
    reqPlotDict = {} 
    
    reqPlotDict[(0,0)] = chListData["channel1"]
    reqPlotDict[(0,1)] = chListData["channel2"]
    reqPlotDict[(0,2)] = chListData["channel3"]
    reqPlotDict[(0,3)] = chListData["channel4"]
    reqPlotDict[(1,0)] = chListData["channel5"]
    reqPlotDict[(1,1)] = chListData["channel6"]
    reqPlotDict[(1,2)] = chListData["channel7"]
    reqPlotDict[(1,3)] = chListData["channel8"]
    
    #define plot title
    fileStr = self.fileName.split('/')
    fileStr = fileStr[-1]
    fileStr = fileStr.split('\\')
    fileStr = fileStr[-1]
 
    #plot required data
    fig, axes = plt.subplots(numRows,numCols,figsize=(14, 6))
    for row in range(0,numRows,1):
      for col in range(0,numCols,1):
        if (row,col) not in reqPlotDict : continue
        reqData = reqPlotDict[(row,col)]
        if "dacList" not in reqData : continue 
        axes[row][col].plot(reqData["dacList"],reqData["meanList"],".")
        axes[row][col].set_xlabel('DAC A Code [DAC]', horizontalalignment='right', x=1.0)
        axes[row][col].set_ylabel('Ch WF Mean [ADC]', horizontalalignment='center', x=1.0)
        axes[row][col].set_title( reqData['ch'] )
        
    plotTitle = "DAC Scan Summary: " + str(fileStr)
    if self.minTimestamp != None:
      plotTitle = plotTitle + ", " + str(self.minTimestamp)
    fig.suptitle(plotTitle, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return

  def processMeasurement(self,measNum=None,measInfo=None):
    #get measurement data objects
    measDataObject = self.getMeasData(measNum,measInfo)
    if measDataObject == None:
      print("Could not parse measurement data objects,",measNum)
      return None
    measNumVal = measNum.split("_")
    measNumVal = int(measNumVal[1])
    measData = measDataObject[0]
    measAttrs = measDataObject[1]
    if "sysComments" not in measAttrs:
      print("ERROR processMeasurement, required field sysComments missing from metadata,",measNum)
      return None
    if ("adc_dacA" not in measAttrs) or ("timestamp" not in measAttrs)  :
      return None
    testID = self.parseSysComments(measNum,measAttrs["sysComments"])  
    if testID == None :
      return None 

    adc_dacA = measAttrs["adc_dacA"]
    timestamp = measAttrs["timestamp"]

    if self.minTimestamp == None:
      self.minTimestamp = timestamp
    elif timestamp < self.minTimestamp :
      self.minTimestamp = timestamp

    testProfile = testID.split("_")
    if len(testProfile) != 3 :
      return None
    testProfile = testProfile[2]

    for chName in measData :
      if testProfile == "B1" :
        if (chName != "channel3") and (chName != "channel5") and (chName != "channel7") :
          continue
      if testProfile == "B2" :
        if  (chName != "channel2") and (chName != "channel4") and (chName != "channel6") :
          continue
      if testProfile == "D" :
        if  (chName != "channel1") and (chName != "channel8") :
          continue
      
      wf = measData[chName]
      vals = self.cv3tbAnalyzeFile.getWaveformValsFrom32BitData(wf)
      if chName not in self.measResults:
        self.measResults[chName] = {}
      if adc_dacA not in self.measResults[chName] :
        self.measResults[chName][adc_dacA] = {}
      self.measResults[chName][adc_dacA] = {"mean": np.mean(vals)}      
    return None
    

  #process processed file data dict
  def processFileData(self):
    if self.runResultsDict == None :
      print("ERROR, no data recovered from file ,exiting")
      return None
    if "results" not in self.runResultsDict:
      print("ERROR, no data recovered from file ,exiting")
      return None
    runData =  self.runResultsDict["results"]
    #loop through dict, find measurements with sysComments metadata
    for cnt, (measNum, measInfo) in enumerate(runData.items()):
      self.processMeasurement(measNum,measInfo)
    return None
    

  def processFile(self):
    print("PROCESS FILE")
    self.cv3tbProcessFile.processFile()
    if self.cv3tbProcessFile.runResultsDict == None:
      print("ERROR, could not process ",self.fileName)
      return None
    
    self.cv3tbProcess32Bit.runResultsDict = self.cv3tbProcessFile.runResultsDict
    self.cv3tbProcessFile.runResultsDict = {}
    self.cv3tbProcess32Bit.chsIn32BitMode = ["channel1","channel2","channel3","channel4","channel5","channel6","channel7","channel8"]
    self.cv3tbProcess32Bit.processFile()
    self.runResultsDict = self.cv3tbProcess32Bit.runResultsDict

    self.processFileData()
    self.plotMeasResults()
    return None

def main():
  print("HELLO, running cv3tbProcessRadTestDacData")
  if len(sys.argv) != 2 :
    print("ERROR, cv3tbProcessRadTestDacData requires filename")
    return
  fileName = sys.argv[1]
  print("FILE",fileName)
  cv3tbProcessRadTestDac = CV3TB_PROCESS_RADTESTDAC(fileName)

  #process the hdf5 file and store info in a dict
  cv3tbProcessRadTestDac.processFile()

  return

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
