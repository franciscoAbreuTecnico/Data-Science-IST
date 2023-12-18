from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from config.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by, plot_multiline_chart, autocorrelation_study, eval_stationarity, plot_components
from numpy import array, arange
from matplotlib.pyplot import show, subplots, plot, legend
from matplotlib.figure import Figure
from config.dslabs_functions import set_chart_labels, plot_line_chart, series_train_test_split, plot_forecasting_eval, plot_forecasting_series
from statsmodels.tsa.stattools import adfuller
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show
from sklearn.linear_model import LinearRegression

file_tag = "covid"
target = "deaths"
timecol: str = "date"
data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

train, test = series_train_test_split(data, trn_pct=0.90)

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(data)).reshape(-1, 1)
tstY = test.to_numpy()

model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression")
savefig(f"forecasting_health/transformation/transformation_images/{file_tag}_smoothing_no_window_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"forecasting_health/transformation/transformation_images/{file_tag}_smoothing_no_window_forecast.png")


data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

data = data.rolling(window=10).mean()


train, test = series_train_test_split(data, trn_pct=0.90)

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(data)).reshape(-1, 1)
tstY = test.to_numpy()

for i in range(10):
    trnY[i] = trnY[10]


model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression")
savefig(f"forecasting_health/transformation/transformation_images/{file_tag}_smoothing_10_window_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"forecasting_health/transformation/transformation_images/{file_tag}_smoothing_10_window_forecast.png")




data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

data = data.rolling(window=25).mean()


train, test = series_train_test_split(data, trn_pct=0.90)

trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(data)).reshape(-1, 1)
tstY = test.to_numpy()

for i in range(25):
    trnY[i] = trnY[25]


model = LinearRegression()
model.fit(trnX, trnY)

prd_trn: Series = Series(model.predict(trnX), index=train.index)
prd_tst: Series = Series(model.predict(tstX), index=test.index)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression")
savefig(f"forecasting_health/transformation/transformation_images/{file_tag}_smoothing_25_window_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Linear Regression",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"forecasting_health/transformation/transformation_images/{file_tag}_smoothing_25_window_forecast.png")
