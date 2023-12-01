from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart, get_variable_types


filename = "dataset_health/data/class_pos_covid.csv"
data: DataFrame = read_csv(filename, sep=',', decimal='.', na_values='')
data = data.dropna()

variables_types: dict[str, list] = get_variable_types(data)
vars = variables_types["numeric"] +  variables_types["symbolic"]
if [] != vars:
    target = "stroke"

    n: int = len(vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(vars)):
        var1: str = vars[i]
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])
    savefig(f"dataset_health/profiling/images/covid_scatter")
    show()
else:
    print("Sparsity class: there are no variables.")
