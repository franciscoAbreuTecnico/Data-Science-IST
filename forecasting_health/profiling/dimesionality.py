from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from config.dslabs_functions import plot_line_chart, HEIGHT

file_tag = "covid"
target = "deaths"
data: DataFrame = read_csv(
    "forecasting_health/data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]
print("Nr. Records = ", series.shape[0])
print("First timestamp", series.index[0])
print("Last timestamp", series.index[-1])

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} weekly {target}",
)
show()
savefig("forecasting_health/profiling/images/covid_weekly_reading.png")