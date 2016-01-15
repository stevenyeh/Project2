import pandas
import pandasql
import ggplot
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import scipy
import scipy.stats
import statsmodels.api as sm
import sys

#Wrangling Subway Data
def num_rainy_days(filename):
    '''
    Run a SQL query on a dataframe of
    weather data.
    '''
    weather_data = pandas.read_csv(filename)
    q = """
    SELECT COUNT(*) FROM weather_data WHERE rain = 1;
    """
    #Execute SQL command against the pandas frame
    rainy_days = pandasql.sqldf(q.lower(), locals())
    return rainy_days

def max_temp_aggregate_by_fog(filename):
    weather_data = pandas.read_csv(filename)
    q = """
    SELECT fog, MAX(maxtempi) FROM weather_data GROUP BY fog;
    """
    foggy_days = pandasql.sqldf(q.lower(), locals())
    return foggy_days

def avg_weekend_temperature(filename):
    weather_data = pandas.read_csv(filename)
    q = """
    SELECT avg(meantempi)
    FROM weather_data
    WHERE cast(strftime('%w', date) as integer) = 0
    OR cast(strftime('%w', date) as integer) = 6
    """
    mean_temp_weekends = pandasql.sqldf(q.lower(), locals())
    return mean_temp_weekends

def avg_min_temperature(filename):
    weather_data = pandas.read_csv(filename)
    q = """
    SELECT avg(mintempi) FROM weather_data WHERE mintempi > 55 AND rain = 1;
    """
    avg_min_temp_rainy = pandasql.sqldf(q.lower(), locals())
    return avg_min_temp_rainy

def fix_turnstile_data(filenames):
    '''
    update each row in the text file so there is only one entry per row.
    '''
    for name in filenames:
        f_in = open(name, 'r')
        f_out = open('updated_' + name, 'w')

        reader_in = csv.reader(f_in, delimiter = ',')
        writer_out = csv.writer(f_out, delimiter = ',')

        for line in reader_in:
            for k in range(0, (len(line)-3)/5):
                line_out = [line[0], line[1], line[2], line[k*5+3], line[k*5+4], line[k*5+5], line[k*5+6], line[k*5+7]]
                writer_out.writerow(line_out)

        f_in.close()
        f_out.close()

def create_master_turnstile_file(filenames, output_file):
    '''
    takes the files in the list filenames, which all have the
    columns 'C/A, UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn, EXITSn', and consolidates
    them into one file located at output_file.  There's one row with the column
    headers, located at the top of the file. The input files do not have column header
    rows of their own.
    '''
    with open(output_file, 'w') as master_file:
        master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
        for filename in filenames:
            with open(filename, 'r') as file_in:
                for row in file_in:
                    master_file.write(row)

def filter_by_regular(filename):
    '''
    reads the csv file located at filename into a pandas dataframe,
    and filters the dataframe to only rows where the 'DESCn' column has the value 'REGULAR'.
    '''
    turnstile_data = pandas.read_csv(filename)
    turnstile_data = pandas.DataFrame(turnstile_data)
    turnstile_data = turnstile_data[turnstile_data.DESCn == 'REGULAR']
    return turnstile_data

def get_hourly_entries(df):
    '''
    This function should change cumulative entry numbers to a count of entries since the last reading
    (i.e., entries since the last row in the dataframe).
       1) Create a new column called ENTRIESn_hourly
       2) Assign to the column the difference between ENTRIESn of the current row
          and the previous row. Any NaN is replaced with 1.
    '''
    shift = df.ENTRIESn.shift(1)
    df['ENTRIESn_hourly'] = df.ENTRIESn - shift
    df['ENTRIESn_hourly'][0] = 1
    shift.fillna(value = 1, inplace = True)
    df.fillna(value = 1, inplace = True)
    return df

def get_hourly_exits(df):
    '''
    same as before, just with exits
    '''
    shift = df.EXITSn.shift(1)
    df['EXITSn_hourly'] = df.EXITSn - shift
    df['EXITSn_hourly'][0] = 0
    shift.fillna(value = 0, inplace = True)
    df.fillna(value = 0, inplace = True)
    return df

def time_to_hour(time):
    '''
    extracts the hour part from the input variable time
    and returns it as an integer.
    '''
    hour = int(time[0:2])
    return hour

def reformat_subway_dates(date):
    '''
    The dates in MTA subway data are formatted in the format month-day-year.
    The dates in weather underground data are formatted year-month-day.
    Takes as its input a date in month-day-year format,
    and returns a date in the year-month-day format.
    '''
    date_formatted = datetime.strftime(datetime.strptime(date, "%m-%d-%y"), "%Y-%m-%d")
    return date_formatted

#Analyzing Subway Data

def entries_histogram(turnstile_weather):
    '''
    Plots two histograms on the same axes to show hourly
    entries when raining vs. when not raining.
    The skewed histograms show that you cannot run the Welch's T test since it assumes normality.
    '''
    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 1]).hist(bins = 200, label = 'Rain') # your code here to plot a historgram for hourly entries when it is raining
    (turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 0]).hist(bins = 200, alpha = 0.5, label = 'Non-Rainy') # your code here to plot a historgram for hourly entries when it is not raining
    plt.title('Rain vs. Non-Rainy Days')
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim([0, 4000])
    return plt

def mann_whitney_plus_means(turnstile_weather):
    '''
    Takes the means and runs the Mann Whitney U-test on the
    ENTRIESn_hourly column in the turnstile_weather dataframe.
    Returns:
        1) the mean of entries with rain
        2) the mean of entries without rain
        3) the Mann-Whitney U-statistic and p-value comparing the number of entries
           with rain and the number of entries without rain
    P-value from test suggests that the distribution of number of entries is statistically different
    between rainy and non rainy days (reject the null)
    '''
    rain = turnstile_weather[turnstile_weather['rain'] == 1]['ENTRIESn_hourly']
    norain = turnstile_weather[turnstile_weather['rain'] == 0]['ENTRIESn_hourly']
    with_rain_mean = np.mean(rain)
    without_rain_mean = np.mean(norain)
    U,p = scipy.stats.mannwhitneyu(rain, norain, use_continuity = False)
    return with_rain_mean, without_rain_mean, U, p

def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    params = results.params[1:]
    intercept = results.params[0]

    return intercept, params

def predictions(dataframe):
    '''
    predict the ridership of
    the NYC subway using linear regression with gradient descent.
    '''
    features = dataframe[['rain', 'precipi', 'Hour', 'fog']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    values = dataframe['ENTRIESn_hourly']

    # Perform linear regression
    intercept, params = linear_regression(features, values)
    predictions = intercept + np.dot(features, params)
    return predictions

def plot_residuals(turnstile_weather, predictions):
    #histogram of the residuals
    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist()
    return plt

def compute_r_squared(data, predictions):
    SST = ((data - np.mean(data)) ** 2).sum()
    SSReg = ((predictions - data) ** 2).sum()
    r_squared = 1 - SSReg / SST
    return r_squared

#Visualizing Subway Data

def plot_weather_data(turnstile_weather):
    turnstile_weather['HOUR'] = turnstile_weather['Hour']
    hour_group = turnstile_weather.groupby('Hour')
    hour_mean = hour_group.aggregate(np.mean)
    plot = ggplot(hour_mean, aes(x = 'HOUR', y = 'ENTRIESn_hourly')) + \
           geom_point() + \
           geom_line() + \
           ggtitle('Average Ridership Based on Hour') + \
           stat_smooth(color = 'red') + \
           xlab('Hour') + \
           ylab('Average Entries')
    pandas.options.mode.chained_assignment = None
    return plot

'''
def plot_weather_data(turnstile_weather):
    plot = ggplot(turnstile_weather, aes(x = 'precipi', y = 'ENTRIESn_hourly')) + \
    geom_point() + \
    geom_line()
    return plot
'''
