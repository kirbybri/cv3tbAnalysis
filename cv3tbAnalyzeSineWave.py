import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import glob
import pickle

#BEGIN CV3TB_ANALYZE_SINEWAVE CLASS
class CV3TB_ANALYZE_SINEWAVE(object):

  #__INIT__#
  def __init__(self, fileName = None):
    self.runResultsDict = None
    self.fileNameDict = None
    self.fileName = fileName
    self.dropInitialSamples = False
    self.numDroppedInitialSamples = 0
    self.applyCuts = False

    #calib constants
    self.mdacWeights = [4288, 4288, 4288, 4288, 4288, 4288, 4288, 4288]
    self.sarWeights = [3584,2048,1024,640,384,256,128,224,128,64,32,24,16,10,6,4,2,1,0.5,0.25]

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

  def SINAD(self,fourier):
    sum2 = 0
    for normBin in fourier:
      if normBin==1: continue
      sum2 += normBin**2
    return -10*np.log10(sum2)

  def ENOB(self,fourier):
    return (self.SINAD(fourier)-1.76)/6.02

  def getFftWaveform(self,vals):
    fft_wf = np.fft.fft(vals)
    fftWf_x = []
    fftWf_y = []
    psd = []
    psd_x = []
    for sampNum,samp in enumerate(fft_wf) :
      if sampNum > float( len(fft_wf) ) / 2. :
        continue
      freqVal = 40. * sampNum/float(len(fft_wf))
      sampVal = np.abs(samp)
      if sampNum == 0 :
        sampVal = 0
      fftWf_x.append(freqVal)
      fftWf_y.append(sampVal)
    if np.max(fftWf_y) <= 0 :
      return psd_x,psd

    fourier_fftWf_y = fftWf_y/np.max(fftWf_y)
    for sampNum,samp in enumerate(fourier_fftWf_y) :
      if sampNum == 0 :
        continue
      else:
        psd_x.append( fftWf_x[sampNum] )
        psd.append( 20*np.log10(samp) )
    sinad = self.SINAD(fourier_fftWf_y)
    enob = self.ENOB(fourier_fftWf_y)
    return psd_x,psd,sinad,enob

  def printEnob(self,chId=None,measNum=None):
    if chId == None or self.runResultsDict == None or measNum == None:
      return None
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    if measNum not in runData :
      return
    measData = runData[measNum]
    if chId not in measData:
      return
    chWf = measData[chId][1:]  #account for first sample is bad
    vals = self.getWaveformValsFrom32BitData(chWf)
    psd_x,psd,sinad,enob = self.getFftWaveform(vals)
    print(measNum,"\t",enob)

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
    if "attrs" in measInfo:
      print(measNum,"\t", measInfo["attrs"] )

    chWf = measData[chId]
    #account for bad samples
    if self.dropInitialSamples :
      chWf = chWf[self.numDroppedInitialSamples:]
    return chWf

  def viewWaveform(self,chId=None,measNum=None):
    chWf = self.getMeasChData(chId,measNum)
    if chWf == None :
      return None
    vals = self.getWaveformValsFrom32BitData(chWf)
    psd_x,psd,sinad,enob = self.getFftWaveform(vals)
    #print("Number of samples ",len(chWf))
    print("MEAS#\t",measNum,"\tMEAN\t",np.mean(vals),"\tRMS\t",np.std(vals),"\tRANGE\t",np.max(vals)-np.min(vals),"\tENOB\t",enob )
    #return
    self.plotVals(measNum,vals,psd_x,psd)

  def printMeasSamples(self,chId=None,measNum=None):
    chWf = self.getMeasChData(chId,measNum)
    if chWf == None :
      return None
    pathSplit = self.fileName.split('/')[-1]
    f = open("data/output_cv3tbAnalyzeSineWave_wfData_" + str(pathSplit) + "_" + str(measNum) + ".txt", "w")
    for samp in chWf:
      f.write( str(samp) + "\n")
    f.close()
    return

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
  cv3tbAnalyzeFile = CV3TB_ANALYZE_SINEWAVE(fileName)
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
