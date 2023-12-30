from config.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart
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

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.preprocessing import StandardScaler
from pandas import Series
from sklearn.base import RegressorMixin
from matplotlib.pyplot import savefig
from config.dslabs_functions import FORECAST_MEASURES, DELTA_IMPROVE, plot_line_chart

from sklearn.base import RegressorMixin

from numpy import mean
from pandas import Series
from sklearn.base import RegressorMixin


class RollingMeanRegressor(RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win
        self.memory: list = []

    def fit(self, X: Series):
        self.memory = X.iloc[-self.win_size :]
        # print(self.memory)
        return

    def predict(self, X: Series):
        estimations = self.memory.tolist()
        for i in range(len(X)):
            new_value = mean(estimations[len(estimations) - self.win_size - i :])
            estimations.append(new_value)
        prd_series: Series = Series(estimations[self.win_size :])
        prd_series.index = X.index
        return prd_series




def rolling_mean_study(train: Series, test: Series, measure: str = "R2"):
    win_size = (3, 5, 10, 15, 20, 25, 30, 40, 50)
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
    best_performance: float = -100000

    yvalues = []
    for w in win_size:
        pred = RollingMeanRegressor(win=w)
        pred.fit(train)
        prd_tst = pred.predict(test)

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (w,)
            best_model = pred
        yvalues.append(eval)

    print(f"Rolling Mean best with win={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        win_size, yvalues, title=f"Rolling Mean ({measure})", xlabel="window size", ylabel=measure, percentage=flag
    )

    return best_model, best_params



def scale_all_dataframe(data: DataFrame) -> DataFrame:
    vars: list[str] = data.columns.to_list()
    transf: StandardScaler = StandardScaler().fit(data)
    df = DataFrame(transf.transform(data), index=data.index)
    df.columns = vars
    return df

measure: str = "R2"
file_tag = 'covid'
timecol = 'date'
target = 'deaths'

data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

ss_diff: Series = data.diff().diff()
ss_diff = scale_all_dataframe(ss_diff)
train, test = series_train_test_split(ss_diff, trn_pct=0.90)


trnX = arange(len(train)).reshape(-1, 1)
trnY = train.to_numpy()
tstX = arange(len(train), len(ss_diff)).reshape(-1, 1)
tstY = test.to_numpy()
trnY[0] = 0
trnY[1] = 0

fig = figure(figsize=(HEIGHT, HEIGHT))
best_model, best_params = rolling_mean_study(train, test)
savefig(f"forecasting_health/forecasting/images/{file_tag}_rollingmean_{measure}_study.png")

params = best_params["params"]
prd_trn: Series = best_model.predict(train)
prd_tst: Series = best_model.predict(test)

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Rolling Mean (win={params[0]})")
savefig(f"forecasting_health/forecasting/images/{file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")

plot_forecasting_series(
    train,
    test,
    prd_tst,
    title=f"{file_tag} - Rolling Mean (win={params[0]})",
    xlabel=timecol,
    ylabel=target,
)
savefig(f"forecasting_health/forecasting/images/{file_tag}_rollingmean_{measure}_forecast.png")
