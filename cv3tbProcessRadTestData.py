import json
import sys
import numpy as np
from math import *
import matplotlib.pyplot as plt
from pathlib import Path

from cv3tbProcessRadTestSineData import CV3TB_PROCESS_RADTESTSINE
from cv3tbProcessRadTestDacData import CV3TB_PROCESS_RADTESTDAC
from cv3tbProcessRadTestAD121Data import CV3TB_PROCESS_RADTESTAD121


from fpdf import FPDF

def testSummary(fileName,results):
  #get base of fileName
  
  fileNamePath = Path(fileName)
  if len( fileNamePath.parts  ) == 0 :
    print("INVALID filename")
    return None
    
  fileName = fileNamePath.parts[-1]
  

  pdf = FPDF(format='letter')
  pdf.add_page()
  pdf.set_font("Arial", 'B', size=20)

  pdf.cell(200, 10, txt="RadBoard Test Summary", ln=2, align='C')

  pdf.set_font("Arial", 'B', size=14)
  pdf.cell(60, 5, txt="File name: " + fileName,ln=2, align='L')
  pdf.ln(4)
  
  #sine result
  text = "Sine Wave Test"
  pdf.cell(200,5,txt=text,align='L',ln=2)
  if "sine_plot" in results:
    if results["sine_plot"] == True :
      pdf.image("sinePlot.png", w=180)
  pdf.ln(4)
  text = "Sine Wave Test FFT Parameters"
  pdf.cell(200,5,txt=text,align='L',ln=2)
  if "sine_table" in results:
    if results["sine_table"] == True :
      pdf.image("sineTable.png", w=180)
  pdf.ln(4)  
  
  pdf.add_page()  
  text = "DAC Test"
  pdf.cell(200,5,txt=text,align='L',ln=2)
  if "dac_plot" in results:
    if results["dac_plot"] == True :
      pdf.image("dacPlot.png", w=180)
  pdf.ln(4)
  text = "AD121 Test"
  pdf.cell(200,5,txt=text,align='L',ln=2)
  if "ad121_plot" in results:
    if results["ad121_plot"] == True :
      pdf.image("ad121Plot.png", w=180)
  pdf.ln(4)
  
  pdf.output('summary_' + fileName + '.pdf','F')


def main():
  print("HELLO, running cv3tbProcessRadTestData")
  if len(sys.argv) != 2 :
    print("ERROR, cv3tbProcessRadTestData requires filename")
    return
  fileName = sys.argv[1]
  print("FILE",fileName)
  cv3tbProcessRadTestSine = CV3TB_PROCESS_RADTESTSINE(fileName)
  cv3tbProcessRadTestDac = CV3TB_PROCESS_RADTESTDAC(fileName)
  cv3tbProcessRadTestAd121 = CV3TB_PROCESS_RADTESTAD121(fileName)

  results = {}
  
  print("PROCESS SINE WAVE TEST")
  cv3tbProcessRadTestSine.processFile()
  results["sine_plot"] = cv3tbProcessRadTestSine.gotResults_sine 
  results["sine_table"] = cv3tbProcessRadTestSine.gotResults_table
  print("PROCESS DAC TEST")
  cv3tbProcessRadTestDac.processFile()
  results["dac_plot"] = cv3tbProcessRadTestDac.gotResults
  print("PROCESS AD121 TEST")
  cv3tbProcessRadTestAd121.processFile()
  results["ad121_plot"] = cv3tbProcessRadTestAd121.gotResults
  
  testSummary(fileName,results)
  
  return

#-------------------------------------------------------------------------
if __name__ == "__main__":
  main()
