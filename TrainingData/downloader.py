"""
The purpose of this script is to download the stock and market data
for training the model.
"""

import subprocess

#Run the stockScrapper.py script to download stock data for:
#Close, Open, High, Low, Volume
subprocess.run(["python", "TrainingData/featuresPy/stockScrapper.py"])


#Run the markets.py script to download market data for SPY and VIX
subprocess.run(["python", "TrainingData/featuresPy/markets.py"])

#Run the insiderbuying.py script to download insider buying data
subprocess.run(["python", "TrainingData/featuresPy/insiderbuying.py"])

#Run the insiderbuying.py script to download insider buying data
subprocess.run(["python", "TrainingData/featuresPy/sentiment.py"])