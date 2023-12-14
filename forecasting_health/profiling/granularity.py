from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from config.dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by

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


deaths_weeks: Series = ts_aggregation_by(series, "W")
figure(figsize=(3 * HEIGHT, HEIGHT ))
plot_line_chart(
    deaths_weeks.index.to_list(),
    deaths_weeks.to_list(),
    xlabel="weeks",
    ylabel=target,
    title=f"{file_tag} weekly mean {target}",
)
show()
savefig("forecasting_health/profiling/images/covid_weekly_reading.png")

deaths_months: Series = ts_aggregation_by(series, "M")
figure(figsize=(3 * HEIGHT, HEIGHT ))
plot_line_chart(
    deaths_months.index.to_list(),
    deaths_months.to_list(),
    xlabel="months",
    ylabel=target,
    title=f"{file_tag} monthly mean {target}",
)
show()
savefig("forecasting_health/profiling/images/covid_monthly_reading.png")