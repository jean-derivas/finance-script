# works in python 3.7

import matplotlib.pyplot as plt
from random import gauss
from matplotlib.dates import drange
import numpy as np
import matplotlib.dates as mdates
import datetime
import argparse


def ploting(dates, slow, shigh):
    """
    plot the two series of data given with same date range
    :param dates: x-axis data (represent time)
    :param slow: represents the worst scenario at the end
    :param shigh: represents the best scenario at the tend
    """
    #  TODO : make the possibility to the user to choose the format in parameter of to keep a rule like that
    if len(slow) < 800:  # almost less than 2 year
        display_freq = mdates.MonthLocator()  # every month
    else:  # more than 2 years
        display_freq = mdates.YearLocator()  # every year

    # formating dates in x-axis. Month and year is enough
    xformat = mdates.DateFormatter('%m/%y')  # format month/year

    # ploting parameters
    fig, ax = plt.subplots()
    plt.plot_date(dates, slow)
    plt.plot_date(dates, shigh)
    ax.xaxis.set_major_locator(display_freq)  # x-axis month/year depending on length of data
    ax.xaxis.set_major_formatter(xformat)
    ax.xaxis.set_tick_params(rotation=45, labelsize=10)
    ax.set_title('Best and worst scenarios at the end of monte carlo simulation')
    ax.set_xlabel('period')
    ax.set_ylabel('return (1 is the intial price)')
    ax.grid(True)
    plt.show()


def distribution(final_scenarios):
    """
    Show the distribution of return at the end of scenarios generated.
    :param final_scenarios: list of return
    """
    fig, ax = plt.subplots()
    ax.hist(final_scenarios, bins=20)  # TODO : make a formula to auto-adapt number of bins
    ax.set_title('Histogram of return at the end of the period')
    ax.set_xlabel('return (1 is the initial price)')
    ax.set_ylabel('occurence number')
    plt.show()


if __name__ == "__main__":

    # TODO : raise error in case of error in parameters (not number, too big value,...)
    # argument processing to get nb_iter, mu and sigma of the normal distribution
    parser = argparse.ArgumentParser(description='Process a demo of monte carlo simulation')
    parser.add_argument('--nb_iter', help='number of monte carlo simulation. default value 1000', default=1000)
    parser.add_argument('--mu', help='mean of the daily return (1.0 is expected to be a neutral expectation 1.05 is a '
                                     '5%% daily return in average. default value = 1.00005', default=1.00005)
    parser.add_argument('--sigma', help='volatility of the daily return. default value 0.001', default=0.001)
    args = parser.parse_args()

    nb_iter = int(args.nb_iter)
    mean = float(args.mu)
    sigma = float(args.sigma)

    # dates
    date1 = datetime.date(2018, 1, 1)
    date2 = datetime.date(2018, 12, 31)
    delta = datetime.timedelta(days=1)
    dates = drange(date1, date2, delta)

    # variable to save scenarios
    low_scenario = []
    high_scenario = []
    lastlow = float("inf")  # biggest value to be sure it will be improved as a lowest value
    lasthigh = -lastlow  # highest value to be sure it will be improved as a highest value
    final_scenarios = []

    # loop to get the worst scenario and the best at the end for nb_iter iteration
    for i in range(nb_iter):
        s = np.ones(len(dates))  # make up some random y values
        i = 0
        while i < len(s):
            s[i] = s[i - 1] * (gauss(mean, sigma))
            i += 1
        if s[i - 1] > lasthigh:
            high_scenario = s
            lasthigh = s[i - 1]
        elif s[i - 1] < lastlow:
            low_scenario = s
            lastlow = s[i - 1]
        final_scenarios.append(s[i - 1])

    # We print the quartile 1 and 3 (to show a 75% probability range)
    final_scenarios.sort()
    quartile1 = np.percentile(final_scenarios, 25)
    quartile3 = np.percentile(final_scenarios, 75)
    print("quartile 1 : " + str(quartile1) + " ; quartile 3 :" + str(quartile3))

    ploting(dates, low_scenario, high_scenario)
    distribution(final_scenarios)
