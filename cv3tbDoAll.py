import json
import sys
import numpy as np
from math import *
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from cv3tbProcessFile import CV3TB_PROCESS_FILE
from cv3tbProcess32BitData import CV3TB_PROCESS_32BITDATA
from cv3tbAnalyzeSineWave import CV3TB_ANALYZE_SINEWAVE
from cv3tbAnalyzeMdacCalRun import CV3TB_ANALYZE_MDACCAL
from cv3tbAnalyzeDacScan import CV3TB_ANALYZE_DACSCAN

def main():
  print("HELLO")
  if len(sys.argv) != 3 :
    print("ERROR, program requires filename and channel name as arguments")
    return
  fileName = sys.argv[1]
  chanName = str(sys.argv[2])

  print("FILE",fileName,"CHAN",chanName)

  print("PROCESS FILE")
  cv3tbProcessFile = CV3TB_PROCESS_FILE(fileName)
  cv3tbProcessFile.limitNumSamples = True
  cv3tbProcessFile.maxNumSamples = 100
  cv3tbProcessFile.processFile()
  #cv3tbProcessFile.outputFile()

  if cv3tbProcessFile.runResultsDict == None :
    return

  print("PROCESS 32 BIT DATA")
  cv3tbProcess32Bit = CV3TB_PROCESS_32BITDATA(fileName)
  cv3tbProcess32Bit.runResultsDict = cv3tbProcessFile.runResultsDict
  cv3tbProcess32Bit.chsIn32BitMode = [chanName]
  cv3tbProcess32Bit.processFile()

  if cv3tbProcess32Bit.runResultsDict == None :
    return
  cv3tbProcess32Bit.outputFile()
  return

  print("ANALYZE DATA")
  if False :
    cv3tbAnalyzeFile = CV3TB_ANALYZE_SINEWAVE(fileName)
    cv3tbAnalyzeFile.runResultsDict = cv3tbProcess32Bit.runResultsDict
    cv3tbAnalyzeFile.dropInitialSamples = True
    cv3tbAnalyzeFile.numDroppedInitialSamples = 1
    for measNum in cv3tbAnalyzeFile.runResultsDict["results"] :
      #cv3tbAnalyzeFile.viewWaveform(chanName,measNum)
      cv3tbAnalyzeFile.printMeasSamples(chanName,measNum)

  if False :
    cv3tbAnalyzeFile = CV3TB_ANALYZE_MDACCAL(fileName)
    cv3tbAnalyzeFile.runResultsDict = cv3tbProcess32Bit.runResultsDict
    cv3tbAnalyzeFile.dropInitialSamples = True
    cv3tbAnalyzeFile.numDroppedInitialSamples = 1
    cv3tbAnalyzeFile.processMdacCalRun(chanName)

  if True :
    cv3tbAnalyzeFile = CV3TB_ANALYZE_DACSCAN(fileName)
    cv3tbAnalyzeFile.runResultsDict = cv3tbProcess32Bit.runResultsDict
    cv3tbAnalyzeFile.plotDacLinearityData(chanName)
    #cv3tbAnalyzeFile.plotMdacBits(chanName)
    #cv3tbAnalyzeFile.printMeasSamples(chanName)

  return

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
