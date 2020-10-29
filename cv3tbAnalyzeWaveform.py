import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import glob
import pickle

from scipy.stats import norm
from scipy.optimize import curve_fit

from numpy import exp, linspace, random

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def gaussian(x, amp, cen, wid):
    return amp * exp(-1.*(x-cen)*(x-cen) / (2*wid*wid))

def close_event():
    plt.close() #timer calls this function after 3 seconds and closes the window 

#BEGIN CV3TB_ANALYZE_WAVEFORM CLASS
class CV3TB_ANALYZE_WAVEFORM(object):

  #__INIT__#
  def __init__(self, fileName = None):
    self.runResultsDict = None
    self.fileNameDict = None
    self.fileName = fileName
    self.dropInitialSamples = True
    self.numDroppedInitialSamples = 1
    self.applyCuts = False
    self.applyDdpuCorr = False
    self.dropOverFlowSamples = False

    #calib constants
    self.mdacWeights = [4288, 4288, 4288, 4288, 4288, 4288, 4288, 4288]
    self.sarWeights = [3584,2048,1024,640,384,256,128,224,128,64,32,24,16,10,6,4,2,1,0.5,0.25]
    self.sampOffsetVal = 0.
    #self.mdacWeights = [4*4288, 4*4288, 4*4288, 4*4288, 4*4288, 4*4288, 4*4288, 4*4288]
    #self.sarWeights = [4*3584,4*2048,4*1024,4*640,4*384,4*256,4*128,4*224,4*128,4*64,4*32,4*24,4*16,4*10,4*6,4*4,4*2,4*1,4*0.5,4*0.25]
    #self.sampOffsetVal = 4400.
    
    #self.mdacWeights = [0,0,0,0,0,0,0,0]
    #self.sarWeights = [0,0,0,0,16384*2,16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2,1]

  def getColutaSampleValue(self,sarBits=[],mdacBits=[]):
    val = 0
    for bitNum in range(len(sarBits)):
      val += self.sarWeights[bitNum]*int(sarBits[bitNum])
    for bitNum in range(len(mdacBits)):
      val += self.mdacWeights[bitNum]*int(mdacBits[bitNum])
    return val

  def getWaveformValsFrom32BitData(self,colutaWf,isMdac=True):
    vals = []
    for samp in colutaWf :
      header = samp[0:2]
      clockCheck = samp[2:4]
      mdacBits = samp[4:12]
      sarBits = samp[12:32]
      sarBitNum = int(sarBits,2)
      if isMdac == False :
        mdacBits = "00000000"
      val = self.getColutaSampleValue(sarBits,mdacBits)
      #print("\t",mdacBits,"\t",sarBits)
      #print("\t",val)
      if self.applyDdpuCorr == True :
        val = int(val /4. - self.sampOffsetVal)
      if self.dropOverFlowSamples == True and val > 32767 :
        val = 32767
        #continue
      if self.dropOverFlowSamples == True and val < 0 :
        val = 0
      
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
    #timer = fig.canvas.new_timer(interval = 1000) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)
    
    #axes[0].plot(vals[0:100],".")
    axes[0].plot(vals,".")
    axes[0].set_xlabel('Sample #', horizontalalignment='right', x=1.0)
    axes[0].set_ylabel('ADC CODE [ADC]', horizontalalignment='left', x=1.0)
    axes[0].set_title("COLUTA WAVEFORM, " + str(measNum) )
    #axes[0].set_xlim(0,1000)
    #axes[0].set_xlim(3600,6400)
    #axes[0].set_xlim(16077,16094)
    #axes[0].set_ylim(17087-50,17087+50)
    axes[1].plot(psd_x,psd,"-")
    axes[1].set_xlabel('Frequency [MHz]', horizontalalignment='right', x=1.0)
    axes[1].set_ylabel('PSD [dB]', horizontalalignment='left', x=1.0)
    axes[1].set_title("")
    #axes[1].set_xlim(0,0.5)
    fig.tight_layout()
    
    #timer.start()
    plt.show()
    #plt.draw()

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
    if False and "attrs" in measInfo:
      print(measNum,"\t", measInfo["attrs"] )

    chWf = measData[chId]
    #account for bad samples
    if self.dropInitialSamples :
      chWf = chWf[self.numDroppedInitialSamples:]
    return chWf

  def viewWaveform(self,chId=None,measNum=None,doPrint=False,doPlot=False):
    chWf = self.getMeasChData(chId,measNum)
    if chWf == None :
      return None
    vals = self.getWaveformValsFrom32BitData(chWf)
    psd_x,psd,sinad,enob = self.getFftWaveform(vals)
    #print("Number of samples ",len(chWf))
    if doPrint:
      print("MEAS#\t",measNum,"\tMEAN\t",np.mean(vals),"\tRMS\t",np.std(vals),"\tMIN\t",np.min(vals),"\tMAX\t",np.max(vals),"\tRANGE\t",np.max(vals)-np.min(vals),"\tENOB\t",enob )
      print("\t", self.runResultsDict["results"][measNum]["attrs"] )
    #return
    if doPlot:
      self.plotVals(measNum,vals,psd_x,psd)
    #return np.mean(vals)
    return np.mean(vals), np.std(vals)

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

  def getAvgDist(self,chId=None):
    if chId == None :
      return
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]

    allSamps = []
    for measNum in runData :
      measInfo = runData[measNum]
      if "data" not in measInfo:
        return None
      measData = measInfo["data"]
      if chId not in measData:
        return None
      if "attrs" not in measInfo:
        return None
      measAttrs = measInfo["attrs"]
      #print(measNum,"\t", measAttrs )
      chWf = self.getMeasChData(chId,measNum)
      if chWf == None :
        continue
      measNumVal = measNum.split("_")
      if len(measNumVal) != 2 :
        continue
      measNumVal = int(measNumVal[1])
      vals = self.getWaveformValsFrom32BitData(chWf)
      wfRms = np.std(vals)
      print(measNumVal,"\t",wfRms)
      for samp in vals:
        allSamps.append(samp)

    allMean = np.mean(allSamps)
    allRms = np.std(allSamps)

    #do fit
    fig, axes = plt.subplots(1,1,figsize=(12, 6))
    n, bins, patches = axes.hist( allSamps, bins = np.arange(int(min(allSamps)), int(max(allSamps))+1, 1), density=False )
    axes.set_xlabel('Sample Value [ADC counts]', horizontalalignment='right', x=1.0)
    axes.set_ylabel('# Samples / ADC code', horizontalalignment='left', x=1.0)
    axes.set_title("Pedestal Sample Distribution")
    #axes.set_xlim(1050,1100)
    #axes.set_ylim(0,50)
    centers = (0.5*(bins[1:]+bins[:-1]))
    pars, cov = curve_fit(gaussian, centers, n, p0=[1.,allMean,allRms])

    #get fit plot
    fit_x = []
    fit_y = []
    for point in centers:
      fit_x.append(point)
      fit_y.append( gaussian(point,pars[0],pars[1],pars[2]) )
    axes.plot( fit_x,fit_y)

    #get chi-square=
    print(n)
    chiSq = 0
    num = 0
    for pointNum,point in enumerate(centers):
      if n[pointNum] == 0 :
        continue
      num = num + 1
      fitVal = gaussian(point,pars[0],pars[1],pars[2])
      err = np.sqrt( n[pointNum] )
      resid = n[pointNum] - fitVal
      chiSq = chiSq + resid*resid/err/err
      #print(point,"\t",n[pointNum],"\t",fitVal,"\t",err)
    #if num > 3:
    #  chiSq = chiSq / float( num - 3 )

    #draw fit label
    textstr = '\n'.join((
      r'$\mu=%.2f\pm%.2f$' % (pars[1], np.sqrt(cov[1,1]), ),
      r'$\sigma=%.2f\pm%.2f$' % (np.abs(pars[2]),np.sqrt(cov[2,2]), ),
      r'$\chi^2 / ndf=%.2f / %d$' % (float(chiSq),float(num), )
    ))          
    axes.text(0.05, 0.95, textstr, transform=axes.transAxes, fontsize=14, verticalalignment='top')

    print( pars )
    print( cov )
    print("TOTAL","\t",allMean,"\t",allRms)

    fig.tight_layout()
    plt.show()

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
  cv3tbAnalyzeFile = CV3TB_ANALYZE_WAVEFORM(fileName)
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
