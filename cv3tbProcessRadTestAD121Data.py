import json
import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
import datetime

from cv3tbProcessFile import CV3TB_PROCESS_FILE
import os
import h5py

#BEGIN CLASS
class CV3TB_PROCESS_RADTESTAD121(object):
  #__INIT__#
  def __init__(self, fileName = None):
    #global objects/variables
    self.fileName = fileName
    self.cv3tbProcessFile = CV3TB_PROCESS_FILE(self.fileName)
    
    self.recMeasNum = []
    self.recChName = []
    self.recItr = []
    self.recAmp = []
    self.recEnob = []
    self.recTimestamp = []
    self.measResults = {}
    self.minTimestamp = None
    self.gotResults = False
    self.plotAfter = True
    
  def processFileData(self):
    if self.fileName == None:
      print("ERROR: no file name supplied")
      return None
    if os.path.isfile(self.fileName) == False :
      print("ERROR: file does not exist ",self.fileName)
      return None
    try:
      self.hdf5File = h5py.File(self.fileName, "r") #file object
    except:
      print("ERROR: couldn't open file",self.fileName)
      return None
      
    #find last timestampt  
    self.lastTimestamp = None
    for measNum in self.hdf5File.keys() :
      #print( "Measurement","\t",measNum)
      meas = self.hdf5File[measNum]
      measAttrs = meas.attrs
      if "timestamp" not in measAttrs : 
        print("NO timestamp")
        continue
      timestamp = measAttrs["timestamp"]
      if self.lastTimestamp == None :
        self.lastTimestamp = timestamp
      if timestamp > self.lastTimestamp :
        self.lastTimestamp = timestamp
    #return None
        
    #loop through measurements, store results in dict
    weights = [2048,1024,512,256,128,64,32,16,8,4,2,1] #generic binary weights for 16-bit words
    for measNum in self.hdf5File.keys() :
      #print( "Measurement","\t",measNum)
      meas = self.hdf5File[measNum]
      measAttrs = meas.attrs
      if "timestamp" not in measAttrs : 
        print("NO timestamp")
        continue
      timestamp = measAttrs["timestamp"]
            
      #compare timestamps
      datetime_timestamp = datetime.datetime.strptime(timestamp, '%y_%m_%d_%H_%M_%S.%f')
      datetime_lastTimestamp = datetime.datetime.strptime(self.lastTimestamp, '%y_%m_%d_%H_%M_%S.%f')
      difference = datetime_lastTimestamp - datetime_timestamp
      difference = difference.total_seconds() / 60.
      #require all timestamps within 30 minutes of last timestamp
      #print(timestamp,"\t",self.lastTimestamp,"\t",datetime_timestamp,"\t",datetime_lastTimestamp,"\t",difference)
      #require all timestamps within 30 minutes of last timestamp
      if (difference > 30) and (self.plotAfter == True) :
        continue
      if (difference <= 30) and (self.plotAfter == False) :
        continue
        
      if self.minTimestamp == None:
        self.minTimestamp = timestamp
      elif timestamp < self.minTimestamp :
        self.minTimestamp = timestamp
            
      if "sysComments" not in measAttrs : continue
      if "dos_dacA" not in measAttrs : continue
      for measType in meas :
        if measType != "ad121" : continue
        sysComments = measAttrs["sysComments"]
        if "FEUG0_FEG0_BEG15" not in sysComments : continue
        measInfo = meas[measType]
        for adc in measInfo :
          adcInfo = measInfo[adc]
          if "raw_data" not in adcInfo : continue
          adcWf = adcInfo["raw_data"]
          vals = np.dot(adcWf, weights) #SAR bits are stored in an int because storing in bytearray took too long, this is silly
          mean = np.mean(vals)
          rms = np.std(vals)
          if adc not in self.measResults :
            self.measResults[adc] = []
          self.measResults[adc].append( {"dos_dacA":measAttrs["dos_dacA"],"mean":mean,"rms":rms} )
    return None

  def plotMeasResults(self):
    #dimensions of summary plots 
    numRows = 2
    numCols = 2

    #dict of required results, organized by plot panel
    chList = ["adc121A","adc121B","adc121C","adc121D"]
    chName = {"adc121A":"adc121A","adc121B":"adc121B","adc121C":"adc121C","adc121D":"adc121D"}
    #chList = ["channel6"]
    chListData = {}
    for ch in chList:
      if ch not in self.measResults :
        chListData[ch] = {}
        continue  
      dacList = []
      meanList = []
      rmsList = []
      for meas in self.measResults[ch] :
        dacList.append( meas["dos_dacA"] )
        meanList.append( meas["mean"] )
        rmsList.append( meas["rms"] )
      chListData[ch] = {'ch':ch,'chName':chName[ch],'dacList':dacList ,'meanList':meanList,'rmsList':rmsList}  
    reqPlotDict = {} 
    
    reqPlotDict[(0,0)] = chListData["adc121A"]
    reqPlotDict[(0,1)] = chListData["adc121B"]
    reqPlotDict[(1,0)] = chListData["adc121C"]
    reqPlotDict[(1,1)] = chListData["adc121D"]
    
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
        #axes[row][col].plot(reqData["dacList"],reqData["meanList"],".")
        axes[row][col].errorbar(x=reqData["dacList"],y=reqData["meanList"],yerr=reqData["rmsList"],fmt=".")
        axes[row][col].set_xlabel('DAC A Code [DAC]', horizontalalignment='right', x=1.0)
        axes[row][col].set_ylabel('Ch WF Mean [ADC]', horizontalalignment='center', x=1.0)
        axes[row][col].set_title( reqData['chName'] )

    plotTitle = "Summary: " + str(fileStr)
    if self.minTimestamp != None :
      plotTitle = plotTitle + ", Time " + self.minTimestamp
    fig.suptitle(plotTitle, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.show()
    
    plt.savefig('ad121Plot.png',
      bbox_inches='tight',
      dpi=150
      )
    self.gotResults = True
    return
    

  def processFile(self):
    print("PROCESS FILE")
    self.processFileData()
    if self.measResults == None :
      return None
    self.plotMeasResults()
    #print( self.measResults )
   
    return None

def main():
  print("HELLO, running cv3tbProcessRadTestDacData")
  if len(sys.argv) != 2 :
    print("ERROR, cv3tbProcessRadTestDacData requires filename")
    return
  fileName = sys.argv[1]
  print("FILE",fileName)
  cv3tbProcessRadTestAd121 = CV3TB_PROCESS_RADTESTAD121(fileName)

  #process the hdf5 file and store info in a dict
  cv3tbProcessRadTestAd121.processFile()

  return

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
