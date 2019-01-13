import os
import matplotlib.pyplot as plt
import mpl_finance as mpf
from matplotlib.dates import date2num
import numpy
import pandas
import datetime

DIR_PATH = os.path.dirname(__file__) or "."
DATA_PATH = "{}/../data".format(DIR_PATH)

def run():
    print("{}/chart".format(DATA_PATH))
    chart_files = os.listdir("{}/chart".format(DATA_PATH))
    print(chart_files)
    for chart_file in chart_files:
        chart_data = pandas.read_csv("{}/chart/{}".format(DATA_PATH, chart_file))
        jpg_name = chart_file.replace("csv", "jpg")
        to_jpg(chart_data, jpg_name)

def to_jpg(chart_data, jpg_name):
    fig = plt.figure()
    ax = plt.subplot()
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    xdate = [
        datetime.datetime.strptime(x, "%Y-%m-%d") for x in chart_data["date"]
    ]
    ohlc = numpy.vstack((date2num(xdate), chart_data[["open", "high", "low", "close"]].values.T)).T
    mpf.candlestick_ohlc(ax, ohlc, width=0.7, colorup='r', colordown='g')
    ax.set_xlim(xdate[0], xdate[-1])
    fig.autofmt_xdate()
    fig.savefig("{}/img/{}".format(DATA_PATH, jpg_name))

if __name__ == "__main__":
    run()