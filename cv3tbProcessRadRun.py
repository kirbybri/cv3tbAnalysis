import json
import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt

from cv3tbProcessRadRunFile import CV3TB_PROCESS_RADRUNFILE
from cv3tbProcess32BitData import CV3TB_PROCESS_32BITDATA
from cv3tbAnalyzeSineWave import CV3TB_ANALYZE_SINEWAVE

#BEGIN CLASS
class CV3TB_PROCESS_RADRUN(object):
  #__INIT__#
  def __init__(self, fileName = None):
    #global objects/variables
    self.fileName = fileName
    self.cv3tbProcessFile = CV3TB_PROCESS_RADRUNFILE(self.fileName)
    self.cv3tbProcess32Bit = CV3TB_PROCESS_32BITDATA(self.fileName)
    self.cv3tbAnalyzeFile = CV3TB_ANALYZE_SINEWAVE(self.fileName)
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

    #print some info about what the program expects
    #print("PROCESSING RAD TEST DATA")
    #print("\tEXPECT ",self.numSampleReq," SAMPLES IN EACH READOUT")    

  #extract required info from sysComments metadata
  def parseSysComments(self,measNum=None,sysComments=None):
    if measNum == None or sysComments == None:
      print("ERROR parseSysComments, invalid input",measNum,sysComments)
      return None
    sysComments = sysComments.split(";")
    if len(sysComments) != self.reqLengthSysComments:
      print("ERROR parseSysComments, metadata field sysComments does not have expected length,",measNum)
      return None
    chName = sysComments[0]
    if chName != "SAR1" and chName != "SAR8" and chName != "MDAC1" and chName != "MDAC2" and chName != "MDAC3" and chName != "MDAC4" :
      print("ERROR parseSysComments, invalid value for chName in sysComments meta-data field,",measNum,chName)
      return None
    freq = sysComments[2]
    freq = freq.split("=")
    if len(freq) != 2 :
      print("ERROR parseSysComments, invalid value for freq value in sysComments meta-data field,",measNum,freq)
      return None
    freq = freq[1]
    amp = sysComments[3]
    amp = amp.split("=")
    if len(amp) != 2 :
      print("ERROR parseSysComments, invalid value for amp value in sysComments meta-data field,",measNum,amp)
      return None
    amp = amp[1]
    itr = sysComments[4]
    return chName,freq,amp,itr

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
    reqPlotDict = {}
    reqPlotDict[(0,0)] = {'ch':"MDAC1",'amp':"0.37",'itr':"itr=1of3",'data':"wf","title":"MDAC1,1Vpp"}
    reqPlotDict[(0,1)] = {'ch':"MDAC2",'amp':"0.37",'itr':"itr=1of3",'data':"wf","title":"MDAC2,1Vpp"}
    reqPlotDict[(0,2)] = {'ch':"MDAC3",'amp':"0.37",'itr':"itr=1of3",'data':"wf","title":"MDAC3,1Vpp"}
    reqPlotDict[(0,3)] = {'ch':"MDAC4",'amp':"0.37",'itr':"itr=1of3",'data':"wf","title":"MDAC4,1Vpp"}
    reqPlotDict[(1,0)] = {'ch':"SAR1",'amp':"0.37",'itr':"itr=1of3",'data':"wf","title":"SAR1,1Vpp"}
    reqPlotDict[(1,1)] = {'ch':"SAR1",'amp':"0.37",'itr':"itr=1of3",'data':"psd","title":"SAR1,1Vpp PSD"}
    reqPlotDict[(1,2)] = {'ch':"SAR8",'amp':"0.37",'itr':"itr=1of3",'data':"wf","title":"SAR8,1Vpp"}
    reqPlotDict[(1,3)] = {'ch':"SAR8",'amp':"0.37",'itr':"itr=1of3",'data':"psd","title":"SAR8,1Vpp PSD"}

    #define plot title
    fileStr = self.fileName.split('/')
    fileStr = fileStr[-1]
    fileStr = fileStr.split('\\')
    fileStr = fileStr[-1]
 
    #plot required data
    fig, axes = plt.subplots(numRows,numCols,figsize=(14, 6))
    for row in range(0,numRows,1):
      for col in range(0,numCols,1):
        reqData = reqPlotDict[(row,col)]
        if reqData['data'] == "wf" :
          vals = []
          if reqData['ch'] in self.measResults :
            if reqData['amp'] in self.measResults[ reqData['ch'] ] :
              if reqData['itr'] in self.measResults[ reqData['ch'] ][ reqData['amp'] ] :
                if reqData['data'] in self.measResults[ reqData['ch'] ][ reqData['amp'] ][ reqData['itr'] ] :
                  vals = self.measResults[ reqData['ch'] ][ reqData['amp'] ][ reqData['itr'] ][ reqData['data'] ]
          if len(vals) > 200 :
            vals = vals[0:200]
          axes[row][col].plot(vals,".")
          axes[row][col].set_xlabel('Sample #', horizontalalignment='right', x=1.0)
          axes[row][col].set_ylabel('ADC CODE [ADC]', horizontalalignment='left', x=1.0)
        if reqData['data'] == "psd" :
          vals_x = []
          vals_y = []
          if reqData['ch'] in self.measResults :
            if reqData['amp'] in self.measResults[ reqData['ch'] ] :
              if reqData['itr'] in self.measResults[ reqData['ch'] ][ reqData['amp'] ] :
                if ( "psd_x" in self.measResults[ reqData['ch'] ][ reqData['amp'] ][ reqData['itr'] ]) and ("psd_y" in self.measResults[ reqData['ch'] ][ reqData['amp'] ][ reqData['itr'] ]):
                  vals_x = self.measResults[ reqData['ch'] ][ reqData['amp'] ][ reqData['itr'] ][ "psd_x" ]
                  vals_y = self.measResults[ reqData['ch'] ][ reqData['amp'] ][ reqData['itr'] ][ "psd_y" ]
          axes[row][col].plot(vals_x,vals_y,".")
          axes[row][col].set_xlabel('Frequency [MHz]', horizontalalignment='right', x=1.0)
          axes[row][col].set_ylabel('PSD [dB]', horizontalalignment='left', x=1.0)
        axes[row][col].set_title( reqData['title'] )
        
    plotTitle = "Summary: " + str(fileStr)
    if len(self.recTimestamp) > 0 :
      plotTitle = plotTitle + ", Time " + self.recTimestamp[0]
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
    #sysComments = measDataObject[1]
    measAttrs = measDataObject[1]
    
    print(measNum)
    print("\t",measNumVal)
    
    if "timestamp" not in measAttrs:
      print("ERROR processMeasurement, required field timestamp missing from metadata,",measNum)
      return None
    if "meas_number" not in measAttrs:
      print("ERROR processMeasurement, required field meas_number missing from metadata,",measNum)
      return None
    if "adc_dacA" not in measAttrs:
      print("ERROR processMeasurement, required field adc_dacA missing from metadata,",measNum)
      return None
    if "adc_dacB" not in measAttrs:
      print("ERROR processMeasurement, required field adc_dacB missing from metadata,",measNum)
      return None

    timestamp = measAttrs["timestamp"]
    meas_number = measAttrs["meas_number"]
    adc_dacA = measAttrs["adc_dacA"]
    adc_dacB = measAttrs["adc_dacB"]

    for meas in measData :
      if 'measType' not in meas :
        continue
      measType = meas["measType"] 
      if measType != "histogram" :
        continue
      #organize histogram data
      chName = meas['channel']
      bin_count = meas['bin_count']
      bin_number = meas['bin_number']
      if measType not in self.measResults:
        self.measResults[measType] = {}
      if chName not in self.measResults[measType] :
        self.measResults[measType][chName] = {}
      if meas_number not in self.measResults[measType][chName] :
        self.measResults[measType][chName][meas_number] = {}
      self.measResults[measType][chName][meas_number] = {"adc_dacA":adc_dacA,"adc_dacB":adc_dacB,"timestamp":timestamp,"bin_count":bin_count,"bin_number":bin_number}
    return None
    
  def getDacVoltage(self,dacCode):
      dacCode = int(dacCode)
      if (dacCode < 0) or ( dacCode > 65535) :
        return 0
      return 2.4 - dacCode*2.4/65535.;
    
  def plotDacVsTime(self):
    vals_x = []
    vals_y = []
    min_xVal = None
    for measType in self.measResults :
      for chName in self.measResults[measType] :
        if chName != "1" : continue
        for meas_number in self.measResults[measType][chName] :
          if "adc_dacA" not in self.measResults[measType][chName][meas_number] :
            continue
          adc_dacA = self.measResults[measType][chName][meas_number]["adc_dacA"]
          adc_dacB = self.measResults[measType][chName][meas_number]["adc_dacB"]
          timestamp = self.measResults[measType][chName][meas_number]["timestamp"]
          
          timestamp = timestamp.split("_")
          second = int(timestamp[1])*30*24*60*60 + int(timestamp[2])*24*60*60 + int(timestamp[3])*60*60 + int(timestamp[4])*60 + float(timestamp[5])
          minute = second / 60.
          hour = second / 60. / 60.
          if min_xVal == None :
            min_xVal = minute
          elif minute < min_xVal :
            min_xVal = minute
          vals_x.append( minute )
          vals_y.append( adc_dacA )
          print(measType,"\t",chName,"\t",meas_number,"\t", adc_dacA ,"\t", adc_dacB)
          
    new_vals_x = [x - min_xVal for x in vals_x]
          
    fig, axes = plt.subplots(figsize=(10, 6))
    axes.plot(new_vals_x,vals_y,".")
    axes.set_xlabel('Time [m]', horizontalalignment='center', x=0.5)
    axes.set_ylabel('DAC code', horizontalalignment='center', x=1.0)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    axes.set_title("DAC code vs Time")

    #fig.suptitle(plotTitle, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
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
    self.runResultsDict = self.cv3tbProcessFile.runResultsDict
    self.processFileData()
    #self.printMeasResults()
    #self.plotMeasResults()
    #self.plotHistResults()
    self.plotDacVsTime()
    return None

def main():
  print("HELLO, running cv3tbProcessRadTestHists")
  if len(sys.argv) != 2 :
    print("ERROR, cv3tbProcessRadTestHists requires filename")
    return
  fileName = sys.argv[1]
  print("FILE",fileName)
  cv3tbProcessRadRun = CV3TB_PROCESS_RADRUN(fileName)

  #process the hdf5 file and store info in a dict
  cv3tbProcessRadRun.processFile()

  return

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
