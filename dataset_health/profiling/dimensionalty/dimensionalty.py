import sys
sys.path.append('.')
sys.path.append('config')

from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart, get_variable_types

filename = "dataset_health/data/class_pos_covid.csv"
data: DataFrame = read_csv(filename, sep=',', decimal='.', na_values='')

images_path: str = "dataset_health/profiling/dimensionalty/images/"

figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"{images_path}covid_records_variables.png", bbox_inches='tight')
show()


mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(25, 6))
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"{images_path}covid_mv.png", bbox_inches='tight')
show()


variable_types: dict[str, list] = get_variable_types(data)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"{images_path}covid_variable_types.png",  bbox_inches='tight')
show()
