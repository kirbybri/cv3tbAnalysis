import h5py
import numpy as np
from math import *
import matplotlib.pyplot as plt
import json
import sys
import pickle
import os.path

class CV3TB_PROCESS_RADRUNFILE(object):

  #__INIT__#
  def __init__(self,fileName=None):
    self.fileName = fileName
    self.hdf5File = None
    self.runResultsDict = None
    self.weights = [32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2,1] #generic binary weights for 16-bit words
    self.maxNumSamples = 10000
    self.limitNumSamples = False

  #analysis process applied to each "measurement"
  def processMeasurement(self, meas):
    #loop over measurement group members, process waveform data
    resultsList = []
    for group_key in meas.keys() :
      if group_key == "DAQ" :
        continue
      if group_key != "histogram" :
        continue
      mysubgroup = meas[group_key] #group object
      for subgroup_key in mysubgroup.keys() :
        mysubsubgroup = mysubgroup[subgroup_key] #group object
        if ("bin_count" not in mysubsubgroup) or ("bin_number" not in mysubsubgroup):
          continue 
        bin_count = mysubsubgroup["bin_count"].value
        bin_number = mysubsubgroup["bin_number"].value
        measType = group_key
        channelNum = subgroup_key
        resultsDict = {'measType' : measType, 'channel' : channelNum, 'bin_count' : bin_count.tolist(), 'bin_number' : bin_number.tolist()}
        resultsList.append( resultsDict )
    return resultsList

  #extract run # from file name
  def getRunNo(self,fileName):
    pathSplit = fileName.split('/')
    nameSplit = pathSplit[-1].split('_')
    if len(nameSplit) < 2 :
      return None
    if nameSplit[0] != 'Run' :
      return None
    return nameSplit[1]

  def getReqAttrs(self,meas=None):
    if meas == None :
      return None
    measAttrs = {}
    for attr in meas.attrs :
      #print(attr,"\t",meas.attrs[attr])
      measAttrs[attr] = meas.attrs[attr]
    return measAttrs

  #run analysis on input file
  def processFile(self):
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
    #check for required attributes
    runNo = self.getRunNo( self.hdf5File.filename )
    if runNo == None :
      print("ERROR: couldn't get run number")

    #define run results
    self.runResultsDict = {}
    self.runResultsDict['file'] = self.hdf5File.filename
    self.runResultsDict['run'] = runNo

    #loop through measurements, store results in dict
    measResultsDict = {}
    for measNum in self.hdf5File.keys() :
      print( "Measurement","\t",measNum)
      meas = self.hdf5File[measNum]
      measData = self.processMeasurement(meas)
      if measData == None :
        print("Missing waveform data, will not process measurement")
        continue
      measAttrs = self.getReqAttrs(meas)
      measResultsDict[measNum] = {'data':measData,'attrs':measAttrs}
  
    #measurement results stored in dict
    self.runResultsDict['results'] = measResultsDict
    self.hdf5File.close()
    return

  #output results dict to json file
  def outputFile(self):
    if self.runResultsDict == None:
      return None
    pathSplit = self.fileName.split('/')[-1]
    #jsonFileName = 'output_cv3tbProcessFile_' + pathSplit + '.json'
    #with open( jsonFileName , 'w') as outfile:
    #  json.dump( self.runResultsDict, outfile, indent=4)
    pickleFileName = 'output_cv3tbProcessFile_' + pathSplit + '.pickle'
    print("Output file ",pickleFileName)
    pickle.dump( self.runResultsDict, open( pickleFileName, "wb" ) )
    return

def main():
  print("HELLO")
  if len(sys.argv) != 2 :
    print("ERROR, program requires filename as argument")
    return
  fileName = sys.argv[1]
  cv3tbProcessFile = CV3TB_PROCESS_RADRUNFILE(fileName)
  cv3tbProcessFile.processFile()
  cv3tbProcessFile.outputFile()

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
