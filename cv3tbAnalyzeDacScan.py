import numpy as np
from math import *
import matplotlib.pyplot as plt
import sys
import glob
import pickle
import statsmodels.api as sm
import scipy.stats

#BEGIN CV3TB_ANALYZE_DACSCAN CLASS
class CV3TB_ANALYZE_DACSCAN(object):

  #__INIT__#
  def __init__(self, fileName = None):
    self.runResultsDict = None
    self.fileNameDict = None
    self.dacDataDict = None
    self.runAvgDataDict = None
    self.fileName = fileName

    #calib constants
    #self.sarWeights = [3584*4,2048*4,1024*4,640*4,384*4,256*4,128*4,224*4,128*4,64*4,32*4,24*4,16*4,10*4,6*4,4*4,2*4,1*4,2,1]
    #self.mdacWeights = [4288*4, 4288*4, 4288*4, 4288*4, 4288*4, 4288*4, 4288*4, 4288*4]
    #self.mdacWeights = [4288, 4288, 4288, 4288, 4288, 4288, 4288, 4288]
    #self.mdacWeights = [4277.251,4283.219,4285.961,4290.093,4290.856,4283.504,4277.665,4280.296] #ASIC 10 ch8 Jan24 MDAC scan weights
    #self.sarWeights = [3584,2048,1024,640,384,256,128,224,128,64,32,24,16,10,6,4,2,1,0.5,0.25]
    self.mdacWeights = [4348.12, 4345.17, 4345.67, 4346.73, 4349.41, 4346.04, 4345.46, 4344.68]
    self.sarWeights = [3584, 2048.65, 1025.31, 641.709, 385.31, 257.153, 128.205, 221.966, 127.491, 63.5096, 32.0504, 23.7532, 15.755, 9.8403, 5.87503, 3.89925, 1.9401, 1.00507, 0.516821, 0.30187]

    #self.lowRun = 4200
    #self.highRun = 62000
    self.lowRun = 0
    self.highRun = 70000
    self.rmsCut = 5/4.
    self.applyCuts = False

  def getColutaSampleValue(self,sarBits=[],mdacBits=[]):
    val = 0
    for bitNum in range(len(sarBits)):
      val += self.sarWeights[bitNum]*int(sarBits[bitNum])
    for bitNum in range(len(mdacBits)):
      val += self.mdacWeights[bitNum]*int(mdacBits[bitNum])
    return val

  def getWaveformVals(self,colutaWf):
    vals = []
    for samp in colutaWf :
      header = samp[0:2]
      clockCheck = samp[2:4]
      mdacBits = samp[4:12]
      sarBits = samp[12:32]
      val = self.getColutaSampleValue(sarBits,mdacBits)
      vals.append(val)
    return vals

  def measureLinearity(self,xs,ys,ysErr,lowLim,upLim):
    if len(xs) < 3 or len(ys) < 3 :
      print("ERROR TOO FEW POINTS")
      return None
    if len(xs) != len(ys) :
      print("ERROR MISMATCHED LENGTHS")
      return None
    xsFit = []
    ysFit = []
    for num in range(0,len(xs),1):
      if xs[num] <= lowLim or xs[num] > upLim :
        continue
      xsFit.append(xs[num])
      ysFit.append(ys[num])
    if len(ysFit ) < 3 :
      print("ERROR TOO FEW POINTS")
      return None   

    xsFit = sm.add_constant(xsFit)
    #model = sm.OLS(ysFit,xsFit)
    model = sm.GLM(ysFit,xsFit)
    results = model.fit()
    if len(results.params) < 2 :
      print("ERROR FIT FAILED")
      return None
    slope = results.params[1]
    intercept = results.params[0]
    slopeErr = results.bse[1]
    interceptErr = results.bse[0]

    #calculate reduced chi-sq
    chiSq = 0
    resid_x = []
    resid_y = []
    resid_yRms = []
    for num in range(0,len(xs),1):
      if xs[num] <= lowLim or xs[num] > upLim :
        continue
      predY = xs[num]*slope + intercept
      resid = ys[num] - predY
      chiSq = chiSq + resid*resid/ysErr[num]/ysErr[num]
      resid_x.append(xs[num])
      resid_y.append(resid)
      resid_yRms.append(ysErr[num])
    chiSq = chiSq / float( len(ysFit ) - 2 )

    print( "SLOPE ", slope , "\tERR ", slopeErr,"\tCHI2 ",chiSq )
    print( results.summary() )
    return slope, intercept, slopeErr, interceptErr, chiSq,resid_x,resid_y,resid_yRms

  def getSarVsDacVals(self,chId=None,runData=None):
    if chId == None or runData == None:
      print("ERROR,getSarVsDacVals invalid inputs")
      return None
    x = []
    y = []
    yRms = []
    for measNum in runData:
      measInfo = runData[measNum]
      if "data" not in measInfo or "attrs" not in measInfo:
        continue
      measData = measInfo["data"]
      measAttrs = measInfo["attrs"]
      if chId not in measData:
        continue
      measNumVal = measNum.split("_")
      measNumVal = int(measNumVal[1])
      dacVal = measNumVal
      if "dac_aval" in measAttrs :
        dacVal = measAttrs["dac_aval"]
      chWf = measData[chId]
      vals = self.getWaveformVals(chWf)
      avgVal = int(np.mean(vals))
      stdDev = float(np.std(vals))
      #apply cuts
      if self.applyCuts == True :
        #if measNumVal < self.lowRun or measNumVal > self.highRun :
        #  continue
        #if stdDev == 0 or stdDev > self.rmsCut :
        if stdDev == 0 :
          continue
      x.append(dacVal)
      y.append(avgVal)
      yRms.append(stdDev)
      #print(measNumVal,"\t",avgVal)
    #end for l
    orderX = np.argsort(x)
    xOrd = np.array(x)[orderX]
    yOrd = np.array(y)[orderX]
    yRmsOrd = np.array(yRms)[orderX]
    return xOrd,yOrd,yRmsOrd

  def plotMdacBits(self,chId=None,plotTitle=""):
    if chId == None:
      return None
    if self.runResultsDict == None:
      print("ERROR, results dict not defined")
      return
    if "results" not in self.runResultsDict:
      print("ERROR, no results in dict")
      return
    runData = self.runResultsDict["results"]

    x_dacVal = []
    y_mdacRange = []
    maxVal = -1
    minVal = 1000000.
    for measNum in runData:
      measInfo = runData[measNum]
      if "data" not in measInfo or "attrs" not in measInfo:
        continue
      measData = measInfo["data"]
      measAttrs = measInfo["attrs"]
      if chId not in measData:
        continue
      measNumVal = measNum.split("_")
      measNumVal = int(measNumVal[1])
      dacVal = measNumVal
      if "dac_aval" in measAttrs :
        dacVal = measAttrs["dac_aval"]
      colutaWf = measData[chId]
      #print("DAC VALUE ", dacVal)
      mdacVals = []
      for sampNum,samp in enumerate(colutaWf) :
        header = samp[0:2]
        clockCheck = samp[2:4]
        mdacBits = samp[4:12]
        sarBits = samp[12:32]
        #print("\t",mdacBits)
        #convert thermometer curve to integer
        mdacCount = 0
        for bit in mdacBits :
          mdacCount = mdacCount + int(bit)
        #print("\t\t",mdacCount)
        mdacVals.append(mdacCount)
        sampVal = self.getColutaSampleValue(sarBits,mdacBits)
        if sampVal < minVal :
          minVal = sampVal
        if sampVal > maxVal :
          maxVal = sampVal
      x_dacVal.append(dacVal)
      y_mdacRange.append(np.mean(mdacVals))
      #end measNum loop

    print("MIN\t",minVal,"\tMAX\t",maxVal,"\tRANGE\t",maxVal-minVal)

    fig, axes = plt.subplots(figsize=(10, 6))
    axes.plot(x_dacVal,y_mdacRange,".")
    axes.set_xlabel('DAC Code', horizontalalignment='right', x=1.0)
    axes.set_ylabel('Avg. MDAC Range', horizontalalignment='center', x=1.0)
    axes.set_title("MDAC Range vs DAC Code")

    fig.suptitle(plotTitle, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return None

  def plotDacLinearityData(self,chId=None,plotTitle=""):
    if chId == None:
      return None
    if self.runResultsDict == None:
      print("ERROR, results dict not defined")
      return
    if "results" not in self.runResultsDict:
      print("ERROR, no results in dict")
      return
    runData = self.runResultsDict["results"]

    x,y,yRms = self.getSarVsDacVals(chId,runData)
    if len(x) != len(y) :
      print("ERROR, plotDacLinearityData couldn't get sample vs DAC values")
      return None
    lineResult = self.measureLinearity(x,y,yRms,self.lowRun,self.highRun)
    X_plotFit = []
    Y_plotFit = []
    resid_x = []
    resid_y = []
    resid_yRms = []
    textstr = ""
    if lineResult != None :
      #slope, intercept, slopeErr, interceptErr, chiSq,resid_x,resid_y,resid_yRms = linResult
      slope = lineResult[0]
      intercept = lineResult[1]
      slopeErr = lineResult[2]
      interceptErr = lineResult[3]
      chiSq = lineResult[4]
      resid_x = lineResult[5]
      resid_y = lineResult[6]
      resid_yRms = lineResult[7]
      X_plotFit = np.linspace(self.lowRun,self.highRun,1000)
      Y_plotFit = X_plotFit*slope + intercept
      textstr = '\n'.join((
        r'$m=%.3f\pm%.3f$' % (slope, slopeErr, ),
        r'$b=%.2f\pm%.2f$' % (intercept,interceptErr, ),
        r'$\chi^2=%.2f$' % (chiSq, )
      ))

    fig, axes = plt.subplots(2,2,figsize=(10, 6))
    axes[0][0].plot(x,y,".")
    axes[0][0].plot(X_plotFit,Y_plotFit,"-")
    axes[0][0].set_xlabel('DAC Code', horizontalalignment='right', x=1.0)
    axes[0][0].set_ylabel('Avg. COLUTA Sample Value [ADC]', horizontalalignment='center', x=1.0)
    axes[0][0].set_title("COLUTA SAmple Value vs DAC Code")
    #axes[0][0].set_xlim(self.lowRun,self.highRun)
    axes[0][0].text(0.05, 0.45, textstr, transform=axes[0][0].transAxes, fontsize=14, verticalalignment='top')

    axes[0][1].plot(x,yRms,".")
    axes[0][1].set_xlabel('DAC Code', horizontalalignment='right', x=1.0)
    axes[0][1].set_ylabel('Measured RMS [ADC]', horizontalalignment='center', x=1.0)
    #axes[0][1].set_xlim(self.lowRun,self.highRun)
    #axes[0][1].set_ylim(0,10)
    axes[0][1].set_title("RMS vs DAC Code")

    #axes[1][0].plot(resid_x,resid_y,".")
    axes[1][0].errorbar(x=resid_x, y=resid_y, yerr=resid_yRms, fmt='.', ecolor='g', capthick=2)
    axes[1][0].set_xlabel('DAC Code', horizontalalignment='right', x=1.0)
    axes[1][0].set_ylabel('Residual [ADC]', horizontalalignment='center', x=1.0)
    #axes[1][0].set_xlim(8000,9000)
    axes[1][0].set_xlim(self.lowRun,self.highRun)
    #axes[1][0].set_ylim(-15,15)
    axes[1][0].set_title("Fit Residuals vs DAC Code")

    if len(yRms) > 1:
      axes[1][1].hist(yRms,bins = np.arange(min(yRms), max(yRms)+1, 0.1))
    axes[1][1].set_xlabel('Waveform RMS', horizontalalignment='right', x=1.0)
    axes[1][1].set_ylabel('Number of Waveforms', horizontalalignment='center', x=1.0)
    axes[1][1].set_title("Waveform RMS Distribution")
    #axes[1][1].set_xlim(0,10)
    fig.suptitle(plotTitle, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return

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
    return psd_x,psd

  def viewDacScanWaveforms(self,chId=None):
    if chId == None:
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    for measNum in runData:
      measData = runData[measNum]
      if chId not in measData:
        continue
      chWf = measData[chId]
      vals = self.getWaveformVals(chWf)

      #apply some cuts
      stdDev = float(np.std(vals))
      if self.applyCuts == True :
        #if stdDev == 0 or stdDev > self.rmsCut :
        if stdDev == 0 :
          continue
      psd_x,psd = self.getFftWaveform(vals)

      fig, axes = plt.subplots(1,2,figsize=(10, 6))
      axes[0].plot(vals,".")
      axes[0].set_xlabel('Sample #', horizontalalignment='right', x=1.0)
      axes[0].set_ylabel('ADC CODE [ADC]', horizontalalignment='left', x=1.0)
      axes[0].set_title("COLUTA WAVEFORM, " + str(measNum) )
      axes[1].plot(psd_x,psd,".")
      axes[1].set_xlabel('Frequency [MHz]', horizontalalignment='right', x=1.0)
      axes[1].set_ylabel('PSD [dB]', horizontalalignment='left', x=1.0)
      axes[1].set_title("")
      fig.tight_layout()
      plt.show()

  def plotBitVsDac(self,chId=None,bitToPlot=None):
    if chId == None or bitToPlot == None:
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    x = []
    y = []
    for measNum in runData:
      measData = runData[measNum]
      if chId not in measData:
        continue
      measNumVal = measNum.split("_")
      measNumVal = int(measNumVal[1])
      chWf = measData[chId]
      bitVals = []
      for samp in chWf :
        bitVal = samp[31-bitToPlot]
        bitVals.append( int(bitVal) )
      avgBitVal = np.mean(bitVals)
      x.append(measNumVal)
      y.append(avgBitVal)

    orderX = np.argsort(x)
    xOrd = np.array(x)[orderX]
    yOrd = np.array(y)[orderX]
    prevAvgBitVal = 0
    foundTransition = False
    for cnt in range(0,len(xOrd),1):
      measNumVal = xOrd[cnt]
      avgBitVal = yOrd[cnt]
      if cnt > 0 and foundTransition == False:
        if avgBitVal > 0.5 and prevAvgBitVal <= 0.5 :
          print(bitToPlot,"\t",measNumVal,"\t",prevAvgBitVal,"\t",avgBitVal)
          foundTransition = True
      prevAvgBitVal = avgBitVal

    fig, axes = plt.subplots(1,1,figsize=(10, 6))
    axes.plot(x,y,".")
    axes.set_xlim(56000,57000)
    axes.set_xlabel('Measurement #', horizontalalignment='right', x=1.0)
    axes.set_ylabel('Avg Bit Value', horizontalalignment='left', x=1.0)
    axes.set_title("Avg Bit Value vs Measurement #, bit # " + str(bitToPlot) )
    fig.tight_layout()
    plt.show()    
    return

  def printMeasSamples(self,chId=None):
    if chId == None:
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    measVals = []
    chSampleStr = []
    for measNum in runData:
      measInfo = runData[measNum]
      if "data" not in measInfo or "attrs" not in measInfo:
        continue
      measData = measInfo["data"]
      measAttrs = measInfo["attrs"]
      if chId not in measData:
        continue
      measNumVal = measNum.split("_")
      measNumVal = int(measNumVal[1])

      chWf = measData[chId]
      vals = self.getWaveformVals(chWf)
      stdDev = float(np.std(vals))
      measVals.append(measNumVal)
      chSampleStr.append(str(measNumVal)+","+",".join(chWf[0:100]))
      #chSampleStr.append(str(measNumVal)+","+",".join(chWf))
      print("\t",str(measNumVal),"\t",",".join(chWf[0:10]) )
    orderMeasVals = np.argsort(measVals)
    measValsOrd = np.array(measVals)[orderMeasVals]
    chSampleStrOrd = np.array(chSampleStr)[orderMeasVals]
    f = open("output_cv3tbAnalyzeDacScan_data.txt", "w")
    for strVal in chSampleStrOrd:
      f.write(strVal+"\n")
    f.close()
    return

  def getMeasAvgSarValForMdacVal(self,chId=None,measNum=None,mdacVal=None):
    if chId == None or measNum == None or mdacVal == None:
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    if measNum not in runData:
      return
    measData = runData[measNum]
    if chId not in measData:
      return
    chWf = measData[chId]
    zeroMdacWeights = [0,0,0,0,0,0,0,0]
    vals = []
    for samp in chWf:
      mdacBits = samp[4:12]
      sarBits = samp[12:32]
      print(samp,"\t",mdacBits,"\t",sarBits)
      sarVal = self.getColutaSampleValue(sarBits,zeroMdacWeights)
      if mdacBits == mdacVal :
        vals.append(sarVal)
    avgVal = float(np.mean(vals))
    print(measNum,"\t",mdacVal,"\t",len(vals),"\t",avgVal)
    return avgVal

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

  ##very specific analysis to look at samples around MDAC transitions
  def analyzeMdacTransition(self,chId=None,mdacBit=None):
    if chId == None or mdacBit==None :
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    bitToPlot = mdacBit + 20
    runData = self.runResultsDict["results"]
    x_measNum = []
    y_mdacBit = []
    y_sarVal = []
    y_sarVal_low = []
    y_sarVal_high = []
    x_sarVal_low_static = []
    x_sarVal_high_static = []
    y_sarVal_low_static = []
    y_sarVal_high_static = []
    yRms_sarVal_low_static = []
    yRms_sarVal_high_static = []
    tempMdacWeights = [0,0,0,0,0,0,0,0]
    mdacLowList = ["00000000","00000001","00000011","00000111","00001111","00011111","00111111","01111111"]
    mdacHighList = ["00000001","00000011","00000111","00001111","00011111","00111111","01111111","11111111"]
    mdacLow = mdacLowList[mdacBit]
    mdacHigh = mdacHighList[mdacBit]
    #print("HERE")
    for measNum in runData:
      measInfo = runData[measNum]
      if "data" not in measInfo or "attrs" not in measInfo:
        return
      measData = measInfo["data"]
      measAttrs = measInfo["attrs"]
      if chId not in measData:
        return      
      measNumVal = measNum.split("_")
      measNumVal = int(measNumVal[1])
      chWf = measData[chId]
      bitVals = []
      sarVals = []
      sarVals_low = []
      sarVals_high = []
      for samp in chWf :
        bitVal = samp[31-bitToPlot]
        mdacBits = samp[4:12]
        sarBits = samp[12:32]
        if mdacBits != mdacLow and mdacBits != mdacHigh :
          continue
        bitVals.append(int(bitVal))
        sarVal = self.getColutaSampleValue(sarBits,tempMdacWeights)
        sarVals.append(sarVal)
        if mdacBits == mdacLow :
          sarVals_low.append(sarVal)
        if mdacBits == mdacHigh :
          sarVals_high.append(sarVal)
      x_measNum.append(measNumVal)
      y_mdacBit.append( np.mean(bitVals) )
      y_sarVal.append( np.mean(sarVals) )
      y_sarVal_low.append( np.mean(sarVals_low) )
      y_sarVal_high.append( np.mean(sarVals_high) )
      if np.mean(bitVals) == 0 :
        x_sarVal_low_static.append(measNumVal)
        y_sarVal_low_static.append( np.mean(sarVals_low) )
        yRms_sarVal_low_static.append( np.std(sarVals_low) )
      if np.mean(bitVals) == 1 :
        x_sarVal_high_static.append(measNumVal)
        y_sarVal_high_static.append( np.mean(sarVals_high) )
        yRms_sarVal_high_static.append( np.std(sarVals_high) )

    orderX = np.argsort(x_measNum)
    xOrd = np.array(x_measNum)[orderX]
    yOrd_mdacBit = np.array(y_mdacBit)[orderX]
    yOrd_sarVal = np.array(y_sarVal)[orderX]
    yOrd_sarVal_low = np.array(y_sarVal_low)[orderX]
    yOrd_sarVal_high = np.array(y_sarVal_high)[orderX]

    orderX = np.argsort(x_sarVal_low_static)
    xOrd_sarVal_low_static = np.array(x_sarVal_low_static)[orderX]
    yOrd_sarVal_low_static = np.array(y_sarVal_low_static)[orderX]
    yRmsOrd_sarVal_low_static = np.array(yRms_sarVal_low_static)[orderX]

    orderX = np.argsort(x_sarVal_high_static)
    xOrd_sarVal_high_static = np.array(x_sarVal_high_static)[orderX]
    yOrd_sarVal_high_static = np.array(y_sarVal_high_static)[orderX]
    yRmsOrd_sarVal_high_static = np.array(yRms_sarVal_high_static)[orderX]

    prevAvgBitVal = 0
    foundTransition = False
    transitionMeas = None
    for cnt in range(0,len(xOrd),1):
      measNumVal = xOrd[cnt]
      avgBitVal = yOrd_mdacBit[cnt]
      if cnt > 0 and foundTransition == False:
        if avgBitVal >= 0.5 and prevAvgBitVal < 0.5 :
          print(bitToPlot,"\t",measNumVal,"\t",prevAvgBitVal,"\t",avgBitVal,"\t",yOrd_sarVal_low[cnt],"\t",yOrd_sarVal_high[cnt],"\t",yOrd_sarVal_low[cnt]-yOrd_sarVal_high[cnt])
          transitionMeas = measNumVal
          foundTransition = True
      prevAvgBitVal = avgBitVal

    plotLowLim = transitionMeas-500
    plotHighLim = transitionMeas+500

    slope_low, intercept_low, slopeErr, interceptErr, chiSq,resid_x,resid_y,resid_yRms = self.measureLinearity(xOrd_sarVal_low_static,yOrd_sarVal_low_static,yRmsOrd_sarVal_low_static,plotLowLim,plotHighLim)
    X_plotFit_low = np.linspace(plotLowLim,plotHighLim,1000)
    Y_plotFit_low = X_plotFit_low*slope_low + intercept_low

    slope_high, intercept_high, slopeErr, interceptErr, chiSq,resid_x,resid_y,resid_yRms = self.measureLinearity(xOrd_sarVal_high_static,yOrd_sarVal_high_static,yRmsOrd_sarVal_high_static,plotLowLim,plotHighLim)
    X_plotFit_high = np.linspace(plotLowLim,plotHighLim,1000)
    Y_plotFit_high = X_plotFit_high*slope_high + intercept_high

    val_low = slope_low*transitionMeas + intercept_low
    val_high = slope_high*transitionMeas + intercept_high
    print("\t",val_low,"\t",val_high,"\t",val_low-val_high)

    fig, axes = plt.subplots(1,2,figsize=(10, 6))
    axes[0].plot(xOrd,yOrd_mdacBit,".")
    if transitionMeas != None :
      axes[0].set_xlim(plotLowLim,plotHighLim)
    axes[0].set_xlabel('Measurement #', horizontalalignment='right', x=1.0)
    axes[0].set_ylabel('Avg Bit Value', horizontalalignment='left', x=1.0)
    axes[0].set_title("Avg Bit Value vs Measurement #, MDAC bit # " + str(mdacBit) )
    axes[1].plot(xOrd,yOrd_sarVal,".",label="Avg SAR Bit Value")
    if transitionMeas != None :
      axes[1].set_xlim(plotLowLim,plotHighLim)
    axes[1].set_xlabel('Measurement #', horizontalalignment='right', x=1.0)
    axes[1].set_ylabel('Avg SAR Bit Value [ADC]', horizontalalignment='left', x=1.0)
    axes[1].set_title("Avg SAR Bit Value vs Measurement #: MDAC bit # " + str(mdacBit) )
    axes[1].plot(xOrd,yOrd_sarVal_low,".",label="MDAC Code " + str(mdacLow))
    axes[1].plot(xOrd,yOrd_sarVal_high,".",label="MDAC Code " + str(mdacHigh))
    #axes[1].plot(xOrd_sarVal_low_static,yOrd_sarVal_low_static,".")
    axes[1].plot(X_plotFit_low,Y_plotFit_low,"-")
    #axes[1].plot(xOrd_sarVal_high_static,yOrd_sarVal_high_static,".")
    axes[1].plot(X_plotFit_high,Y_plotFit_high,"-")
    axes[1].legend()
    fig.tight_layout()
    plt.show()    
    return

  def viewMeasWaveform(self,chId=None,measNum=None):
    if chId == None or measNum == None:
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]
    if measNum not in runData:
      return
    measInfo = runData[measNum]
    if "data" not in measInfo or "attrs" not in measInfo:
      return
    measData = measInfo["data"]
    measAttrs = measInfo["attrs"]
    if chId not in measData:
      return
    chWf = measData[chId]

    print(chWf)

    x = []
    y = []
    x_same = []
    y_same = []
    x_diff = []
    y_diff = []

    prevMdacBits = None
    for sampNum,samp in enumerate(chWf) :
      mdacBits = samp[4:12]
      sarBits = samp[12:32]
      sarVal = self.getColutaSampleValue(sarBits,[0,0,0,0,0,0,0,0])
      x.append(sampNum)
      y.append(sarVal)
      if mdacBits == prevMdacBits or sampNum == 0:
        x_same.append(sampNum)
        y_same.append(sarVal)
      else :
        x_diff.append(sampNum)
        y_diff.append(sarVal)
      prevMdacBits = mdacBits

    fig, axes = plt.subplots(1,1,figsize=(10, 6))
    #axes.plot(x,y,".")
    axes.plot(x_same,y_same,".",label="Sample with no MDAC transition", markersize=16)
    axes.plot(x_diff,y_diff,".",label="Sample with MDAC transition", markersize=16)
    axes.set_xlabel('Sample #', horizontalalignment='right', x=1.0)
    axes.set_ylabel('Sample Value Using SAR Bits Only', horizontalalignment='left', x=1.0)
    axes.set_title("COLUTA Sample Waveform, SAR Bit Values Only, " + str(measNum) )
    axes.set_ylim(2700,2900)
    #axes.set_ylim(6050,6150)
    #axes.legend(fontsize=12)
    fig.tight_layout()
    plt.show()    
    return

  def analyzeMdacTransitionEffect(self,chId=None):
    if chId == None:
      return None
    if self.runResultsDict == None:
      return
    if "results" not in self.runResultsDict:
      return
    runData = self.runResultsDict["results"]

    x_measNum = []
    y_offset = []
    #loop over measurements
    for measNum in runData:
      measData = runData[measNum]
      if chId not in measData:
        continue
      chWf = measData[chId]
      measNumVal = measNum.split("_")
      measNumVal = int(measNumVal[1])
      x_same = []
      samp_same = []
      x_diff = []
      samp_diff = []
      prevMdacBits = None
      for sampNum,samp in enumerate(chWf) :
        mdacBits = samp[4:12]
        sarBits = samp[12:32]
        sarVal = self.getColutaSampleValue(sarBits,[0,0,0,0,0,0,0,0])
        if mdacBits == prevMdacBits:
          samp_same.append(sarVal)
          x_same.append(sampNum)
        elif sampNum > 0:
          samp_diff.append(sarVal)
          x_diff.append(sampNum)
        prevMdacBits = mdacBits
      if len(samp_diff) == 0 : #require some transition samples
        continue
      if len(samp_same) / len(chWf ) < 0.9 : #require most samples to be in same MDAC range
        continue

      baseVal = np.mean(samp_same)
      x_offset = []
      samp_offset = []
      for sampNum,samp in enumerate(samp_diff) :
        if abs(samp - baseVal) > 1000 :
          continue
        x_offset.append( x_diff[sampNum] )
        samp_offset.append(samp)
      offsetVal = np.mean(samp_offset)
      x_measNum.append(measNumVal)
      y_offset.append( offsetVal - baseVal )

      print(measNum,"\t", baseVal ,"\t",offsetVal,"\t", offsetVal - baseVal)
      continue
      #continue

      fig, axes = plt.subplots(1,1,figsize=(10, 6))
      axes.plot(x_same,samp_same,".",label="No MDAC transition")
      axes.plot(x_offset,samp_offset,".",label="MDAC transition")
      axes.set_xlabel('Sample #', horizontalalignment='right', x=1.0)
      axes.set_ylabel('Sample Value Using SAR Bits Only', horizontalalignment='left', x=1.0)
      axes.set_title("COLUTA Sample Waveform, SAR Bit Values Only, " + str(measNum) )
      axes.legend()
      #axes.set_ylim(baseVal - 200, baseVal + 200 )
      fig.tight_layout()
      plt.show()    
      #end meas loop

    fig, axes = plt.subplots(1,2,figsize=(10, 6))
    axes[0].plot(x_measNum,y_offset,".")
    axes[0].set_xlabel('Measurement #', horizontalalignment='right', x=1.0)
    axes[0].set_ylabel('Sample Offset Due to MDAC Transition', horizontalalignment='left', x=1.0)
    axes[0].set_title("Sample Offsets due to MDAC Transition vs Measurement #" )
    #axes[1].hist(y_offset,bins = np.arange(min(y_offset), max(y_offset)+1))
    axes[1].hist(y_offset)
    axes[1].set_xlabel('Sample Offset Due to MDAC Transition', horizontalalignment='right', x=1.0)
    axes[1].set_ylabel('Number of Samples', horizontalalignment='left', x=1.0)
    axes[1].set_title("Sample Offset due to MDAC Transition Distribution" )
    #axes.set_ylim(baseVal - 200, baseVal + 200 )
    fig.tight_layout()
    plt.show()   

    return

  def analyzeDacScanDacDataFile(self):
    if self.fileName == None :
      print("ERROR no input file specified")
      return None

    measNum = []
    dacAVolts = []
    dacBVolts = []
    diffVolts = []
    diffVoltsRms = []
    with open(self.fileName) as fp:
     for cnt, line in enumerate(fp):
       if cnt == 0:
         continue
       #if cnt == 2:
       #  break
       line = line.split("\t")
       #print(line)
       dacAVolt = float(line[0])*1000.
       dacBVolt = float(line[1])*1000.
       measNum.append(cnt-1)
       dacAVolts.append(dacAVolt)
       dacBVolts.append(dacBVolt)
       diffVolts.append(dacBVolt-dacAVolt)
       diffVoltsRms.append(0.00005*1000.)

    slope, intercept, slopeErr, interceptErr, chiSq,resid_x,resid_y,resid_yRms = self.measureLinearity(measNum,diffVolts,diffVoltsRms,self.lowRun,self.highRun)
    X_plotFit = np.linspace(self.lowRun,self.highRun,1000)
    Y_plotFit = X_plotFit*slope + intercept

    textstr = '\n'.join((
      r'$m=%.4f\pm%.4f$' % (slope, slopeErr, ),
      r'$b=%.2f\pm%.2f$' % (intercept,interceptErr, ),
      r'$\chi^2=%.2f$' % (chiSq, )
    ))

    fig, axes = plt.subplots(1,2,figsize=(10, 6))
    axes[0].plot(measNum,dacAVolts,".",label="DAC A voltage")
    axes[0].plot(measNum,dacBVolts,".",label="DAC B voltage")
    axes[0].plot(measNum,diffVolts,".",label="Differential voltage")
    axes[0].plot(X_plotFit,Y_plotFit,".",label="Differential voltage fit")
    axes[0].set_xlabel('Measurement #', horizontalalignment='right', x=1.0)
    axes[0].set_ylabel('DAC voltage [mV]', horizontalalignment='left', x=1.0)
    axes[0].set_title("DAC voltage vs Measurement #" )
    axes[0].text(0.05, 0.45, textstr, transform=axes[0].transAxes, fontsize=14, verticalalignment='top')
    #axes.set_ylim(baseVal - 200, baseVal + 200 )
    axes[0].legend()
    
    axes[1].plot(resid_x,resid_y,".",label="Differential Voltage Fit Residual")
    axes[1].set_xlabel('Measurement #', horizontalalignment='right', x=1.0)
    axes[1].set_ylabel('Fit Residual [mV]', horizontalalignment='left', x=1.0)
    axes[1].set_title("Fit Residual vs Measurement #" )
    fig.tight_layout()
    plt.show()   

    return

def main():
  if len(sys.argv) != 2 :
    print("ERROR, program requires filename argument")
    return
  fileName = sys.argv[1]
  cv3tbAnalyzeFile = CV3TB_ANALYZE_DACSCAN(fileName)
  cv3tbAnalyzeFile.openFile()
  #cv3tbAnalyzeFile.analyzeDacScanDacDataFile() # don't run openFile first for this
  #cv3tbAnalyzeFile.dumpFile()
  #for now need to specify which channels to analyze
  #cv3tbAnalyzeFile.viewDacScanWaveforms("channel1")
  #cv3tbAnalyzeFile.plotDacLinearityData("channel7")
  cv3tbAnalyzeFile.viewMeasWaveform("channel7","Measurement_48889")
  #cv3tbAnalyzeFile.analyzeMdacTransitionEffect("channel8")
  #cv3tbAnalyzeFile.plotDacLinearityData("channel7","")
  #cv3tbAnalyzeFile.printMeasSamples("channel7")
  #cv3tbAnalyzeFile.plotBitVsDac("channel8",18)
  #for bitToPlot in range(0,20,1):
  #  cv3tbAnalyzeFile.plotBitVsDac("channel8",bitToPlot)
  #scan MDAC bits
  #for bitToPlot in range(20,28,1):
  #  cv3tbAnalyzeFile.plotBitVsDac("channel8",bitToPlot)
  #cv3tbAnalyzeFile.plotBitVsDac("channel8",20)
  #cv3tbAnalyzeFile.checkBitRange("channel8")
  #for bitToPlot in range(0,8,1):
  #  cv3tbAnalyzeFile.analyzeMdacTransition(chId="channel7",mdacBit=bitToPlot)
#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
