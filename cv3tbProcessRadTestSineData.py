import json
import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
import datetime

from cv3tbProcessFile import CV3TB_PROCESS_FILE
from cv3tbProcess32BitData import CV3TB_PROCESS_32BITDATA
from cv3tbAnalyzeSineWave import CV3TB_ANALYZE_SINEWAVE

#BEGIN CLASS
class CV3TB_PROCESS_RADTESTSINE(object):
  #__INIT__#
  def __init__(self, fileName = None):
    #global objects/variables
    self.fileName = fileName
    self.cv3tbProcessFile = CV3TB_PROCESS_FILE(self.fileName)
    self.cv3tbProcess32Bit = CV3TB_PROCESS_32BITDATA(self.fileName)
    self.cv3tbAnalyzeFile = CV3TB_ANALYZE_SINEWAVE(self.fileName)
    self.runResultsDict = None
    self.numSampleReq = 2048
    self.numSampleSkip = 0
    self.reqLengthSysComments = 6
    self.gotResults_sine = False
    self.gotResults_table = False

    self.recMeasNum = []
    self.recChName = []
    self.recItr = []
    self.recAmp = []
    self.recEnob = []
    self.recTimestamp = []
    self.measResults = {}
    self.lastTimestamp = None

    #print some info about what the program expects
    #print("PROCESSING RAD TEST DATA")
    #print("\tEXPECT ",self.numSampleReq," SAMPLES IN EACH READOUT")    

  #extract required info from sysComments metadata
  def parseSysComments(self,measNum=None,sysComments=None):
    if measNum == None or sysComments == None:
      #print("ERROR parseSysComments, invalid input",measNum,sysComments)
      return None
    sysComments = sysComments.split(";")
    if len(sysComments) != self.reqLengthSysComments:
      #print("ERROR parseSysComments, metadata field sysComments does not have expected length,",measNum)
      return None
    chName = sysComments[0]
    if chName != "SAR1" and chName != "SAR8" and chName != "MDAC1" and chName != "MDAC2" and chName != "MDAC3" and chName != "MDAC4" :
      #print("ERROR parseSysComments, invalid value for chName in sysComments meta-data field,",measNum,chName)
      return None
    freq = sysComments[2]
    freq = freq.split("=")
    if len(freq) != 2 :
      #print("ERROR parseSysComments, invalid value for freq value in sysComments meta-data field,",measNum,freq)
      return None
    freq = freq[1]
    amp = sysComments[3]
    amp = amp.split("=")
    if len(amp) != 2 :
      #print("ERROR parseSysComments, invalid value for amp value in sysComments meta-data field,",measNum,amp)
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
    """
    orderMeasNum = np.argsort(self.recMeasNum)
    recMeasNumOrd = np.array(self.recMeasNum)[orderMeasNum]
    recChNameOrd = np.array(self.recChName)[orderMeasNum]
    recItrOrd = np.array(self.recItr)[orderMeasNum]
    recAmpOrd = np.array(self.recAmp)[orderMeasNum]
    recEnobOrd = np.array(self.recEnob)[orderMeasNum]
    for num in range(0,len(recMeasNumOrd),1):
      print("Measurement:\t",recMeasNumOrd[num],"\tCh:",recChNameOrd[num],"\tIter:",recItrOrd[num],"\tAmp:",recAmpOrd[num],"\tENOB:",recEnobOrd[num])
    return
    """
    #elf.measResults[chName][amp][itr] = {"wf":vals,"psd_x":psd_x,"psd_y":psd,"enob":enob,"sinad":sinad,"sfdr":sfdr,"snr":snr}

  def makeTable(self):
    reqInfo = []
    reqInfo.append( {"ch":"MDAC1","amp":"3.5","itr":"itr=1of3","enob":"enob"} )
    reqInfo.append( {"ch":"MDAC2","amp":"3.5","itr":"itr=1of3","enob":"enob"} )
    reqInfo.append( {"ch":"MDAC3","amp":"3.5","itr":"itr=1of3","enob":"enob"} )
    reqInfo.append( {"ch":"MDAC4","amp":"3.5","itr":"itr=1of3","enob":"enob"} )
    reqInfo.append( {"ch":"SAR1","amp":"10.0","itr":"itr=1of3","enob":"enob"} )
    reqInfo.append( {"ch":"SAR8","amp":"10.0","itr":"itr=1of3","enob":"enob"} )
    for info in reqInfo :
      if info["ch"] not in self.measResults : return None
      if info["amp"] not in self.measResults[ info["ch"] ] : return None
      if info["itr"] not in self.measResults[ info["ch"] ][ info["amp"] ] : return None
      if "enob" not in self.measResults[ info["ch"] ][ info["amp"] ][ info["itr"] ] : return None

    #define table
    row0 = ["Amp [V]","ENOB"                                               ,"SFDR","SNR"]
    row1 = ["MDAC1","3.5", round(self.measResults["MDAC1"]["3.5"]["itr=1of3"]["enob"],1),round(self.measResults["MDAC1"]["3.5"]["itr=1of3"]["sfdr"],1),round(self.measResults["MDAC1"]["3.5"]["itr=1of3"]["snr"],1) ]
    row2 = ["MDAC2","3.5", round(self.measResults["MDAC2"]["3.5"]["itr=1of3"]["enob"],1),round(self.measResults["MDAC2"]["3.5"]["itr=1of3"]["sfdr"],1),round(self.measResults["MDAC2"]["3.5"]["itr=1of3"]["snr"],1) ]
    row3 = ["MDAC3","3.5", round(self.measResults["MDAC3"]["3.5"]["itr=1of3"]["enob"],1),round(self.measResults["MDAC3"]["3.5"]["itr=1of3"]["sfdr"],1),round(self.measResults["MDAC3"]["3.5"]["itr=1of3"]["snr"],1) ]
    row4 = ["MDAC4","3.5", round(self.measResults["MDAC4"]["3.5"]["itr=1of3"]["enob"],1),round(self.measResults["MDAC4"]["3.5"]["itr=1of3"]["sfdr"],1),round(self.measResults["MDAC4"]["3.5"]["itr=1of3"]["snr"],1) ]
    row5 = ["SAR1","10.0", round(self.measResults["SAR1"]["10.0"]["itr=1of3"]["enob"],1),round(self.measResults["SAR1"]["10.0"]["itr=1of3"]["sfdr"],1),round(self.measResults["SAR1"]["10.0"]["itr=1of3"]["snr"],1) ]
    row6 = ["SAR8","10.0", round(self.measResults["SAR8"]["10.0"]["itr=1of3"]["enob"],1),round(self.measResults["SAR8"]["10.0"]["itr=1of3"]["sfdr"],1),round(self.measResults["SAR8"]["10.0"]["itr=1of3"]["snr"],1) ]
    
    data = [ row0,row1,row2,row3,row4,row5,row6 ]
    
    # Pop the headers from the data array
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]
    
    cell_text = []
    for row in data:
      cell_text.append(row)
      
    title_text = 'FFT Results'
    #footer_text = 'June 24, 2020'
    #fig_background_color = 'skyblue'
    #fig_border = 'steelblue' 
    
    # Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
    plt.figure(linewidth=2,
           #edgecolor=fig_border,
           #facecolor=fig_background_color,
           tight_layout={'pad':1},
           figsize=(5,2)
          )
      
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      #rowColours=rcolors,
                      rowLoc='right',
                      #colColours=ccolors,
                      colLabels=column_headers,
                      loc='center')
                      
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Hide axes border
    plt.box(on=None)
                      
    #plt.show()
    plt.savefig('sineTable.png',
            bbox_inches='tight',
            dpi=150
            )
    self.gotResults_table = True

  def plotMeasResults(self):
    #dimensions of summary plots 
    numRows = 2
    numCols = 4

    #dict of required results, organized by plot panel
    reqPlotDict = {}
    reqPlotDict[(0,0)] = {'ch':"MDAC1",'amp':"3.5",'itr':"itr=1of3",'data':"wf","title":"MDAC1,1Vpp"}
    reqPlotDict[(0,1)] = {'ch':"MDAC2",'amp':"3.5",'itr':"itr=1of3",'data':"wf","title":"MDAC2,1Vpp"}
    reqPlotDict[(0,2)] = {'ch':"MDAC3",'amp':"3.5",'itr':"itr=1of3",'data':"wf","title":"MDAC3,1Vpp"}
    reqPlotDict[(0,3)] = {'ch':"MDAC4",'amp':"3.5",'itr':"itr=1of3",'data':"wf","title":"MDAC4,1Vpp"}
    reqPlotDict[(1,0)] = {'ch':"SAR1",'amp':"10.0",'itr':"itr=1of3",'data':"wf","title":"SAR1,1Vpp"}
    reqPlotDict[(1,1)] = {'ch':"SAR1",'amp':"10.0",'itr':"itr=1of3",'data':"psd","title":"SAR1,1Vpp PSD"}
    reqPlotDict[(1,2)] = {'ch':"SAR8",'amp':"10.0",'itr':"itr=1of3",'data':"wf","title":"SAR8,1Vpp"}
    reqPlotDict[(1,3)] = {'ch':"SAR8",'amp':"10.0",'itr':"itr=1of3",'data':"psd","title":"SAR8,1Vpp PSD"}

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
    #plt.show()
    #plt.plot()
    plt.savefig('sinePlot.png',
      bbox_inches='tight',
      dpi=150
      )
    self.gotResults_sine = True
    
    #plt.savefig(str(fileStr) + ".png")
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
    if "timestamp" not in measAttrs:
      print("ERROR processMeasurement, required field timestamp missing from metadata,",measNum)
      return None
    if "sysComments" not in measAttrs:
      print("ERROR processMeasurement, required field sysComments missing from metadata,",measNum)
      return None
    timestamp = measAttrs["timestamp"]
    sysComments = measAttrs["sysComments"]

    #compare timestamps
    #print(timestamp,"\t",self.lastTimestamp)
    datetime_timestamp = datetime.datetime.strptime(timestamp, '%y_%m_%d_%H_%M_%S.%f')
    datetime_lastTimestamp = datetime.datetime.strptime(self.lastTimestamp, '%y_%m_%d_%H_%M_%S.%f')
    difference = datetime_lastTimestamp - datetime_timestamp
    difference = difference.total_seconds() / 60.
    #require all timestamps within 30 minutes of last timestamp
    if difference > 30 :
      return None


    #parse the sysComments metadata
    sysCommentData = self.parseSysComments(measNum,sysComments)
    if sysCommentData == None:
      #print("Could not parse sysComments metadata,",measNum)
      return None
    chName = sysCommentData[0]
    freq = sysCommentData[1]
    amp = sysCommentData[2]
    itr = sysCommentData[3]

    #process 32-bit mode data
    #print("Process 32-bit data for ",measNum,"\tchannel",chName)
    self.cv3tbProcess32Bit.chsIn32BitMode = [chName]
    ch32BitData = self.cv3tbProcess32Bit.getMeasCh32Bit(measData)
    if chName not in ch32BitData :
      print("Could not get 32-bit data,",measNum,chName)
      return None
    if len(ch32BitData[chName]) != self.numSampleReq :
      print("32-bit data record is not expected length,",measNum,chName,len(ch32BitData[chName]) )
      return None

    #plot 32-bit data
    chWf = ch32BitData[chName][self.numSampleSkip:]  #drop initial number of samples
    isMdac = True
    if (chName == "channel1") or (chName == "channel8") :
      isMdac = False
    vals = self.cv3tbAnalyzeFile.getWaveformValsFrom32BitData(chWf,isMdac)
    psd_x,psd,sinad,enob,sfdr,snr = self.cv3tbAnalyzeFile.getFftWaveform(vals)

    self.recMeasNum.append(measNumVal)
    self.recChName.append(chName)
    self.recItr.append(itr)
    self.recAmp.append(amp)
    self.recEnob.append(enob)
    self.recTimestamp.append(timestamp)

    if chName not in self.measResults:
      self.measResults[chName] = {}
    if amp not in self.measResults[chName] :
      self.measResults[chName][amp] = {}
    self.measResults[chName][amp][itr] = {"wf":vals,"psd_x":psd_x,"psd_y":psd,"enob":enob,"sinad":sinad,"sfdr":sfdr,"snr":snr}

    #debug stuff
    #print("MEAS#",measNum,"\tCH",chName,"\tITR",itr,"\tAmp",amp,"\tMEAN",np.mean(vals),"\tRMS",np.std(vals),"\tRANGE",np.max(vals)-np.min(vals),"\tENOB",enob )
    #print(ch32BitData[chName][0:20])
    #for sampNum in range(0,20,1):
    #  print(ch32BitData[chName][sampNum][0:4],"\t",ch32BitData[chName][sampNum][4:12],"\t",ch32BitData[chName][sampNum][12:32])
    #cv3tbAnalyzeFile.plotVals(measNum,vals,psd_x,psd)
    #f = open("data/output_cv3tbProcessRadTestData_wfData_"+ str(chName) + "_" + str(measNum)  + ".txt", "w")
    #for samp in chWf:
    #  f.write( str(samp) + "\n")
    #f.close()
    return None
    
  def findLastTimestamp(self,measNum=None,measInfo=None):
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
    if "timestamp" not in measAttrs:
      print("ERROR processMeasurement, required field timestamp missing from metadata,",measNum)
      return None
    timestamp = measAttrs["timestamp"]   
    if self.lastTimestamp == None :
      self.lastTimestamp = timestamp
    if timestamp > self.lastTimestamp :
      self.lastTimestamp = timestamp
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
    #identify last time stamp
    for cnt, (measNum, measInfo) in enumerate(runData.items()):
      self.findLastTimestamp(measNum,measInfo)
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
    self.plotMeasResults()
    self.makeTable()
    return None

def main():
  print("HELLO, running cv3tbProcessRadTestData")
  if len(sys.argv) != 2 :
    print("ERROR, cv3tbProcessRadTestData requires filename")
    return
  fileName = sys.argv[1]
  print("FILE",fileName)
  cv3tbProcessRadTestSine = CV3TB_PROCESS_RADTESTSINE(fileName)

  #process the hdf5 file and store info in a dict
  cv3tbProcessRadTestSine.processFile()

  return

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
