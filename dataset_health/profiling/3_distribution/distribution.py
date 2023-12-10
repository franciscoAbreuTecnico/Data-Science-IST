import sys
sys.path.append('.')
sys.path.append('config')
from pandas import Series, read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show, subplots
from matplotlib.figure import Figure
from numpy import ndarray
from dslabs_functions import (
    HEIGHT,
    CLASS_EVAL_METRICS,
    count_outliers,
    define_grid,
    get_variable_types,
    plot_bar_chart,
    plot_multibar_chart,
    set_chart_labels,
    plot_multi_scatters_chart,
    run_NB,
    run_KNN
)
import matplotlib.pyplot as plt


filename = "dataset_health/data/class_pos_covid.csv"
data: DataFrame = read_csv(filename, sep=',', decimal='.', na_values='')


# Boxplot of all numeric variables
variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
if [] != numeric:
    # Create the boxplot
    data[numeric].boxplot(rot=45)
    plt.title('Global Boxplot of Numeric Variables', fontsize=7)
    savefig(f"dataset_health/profiling/distribution/images/covid_global_boxplot.png", bbox_inches='tight')
    show()
else:
    print("There are no numeric variables.")


# single Boxplot per numeric variable
numeric: list[str] = variables_types["numeric"]

if [] != numeric:
    rows: int
    cols: int
    rows, cols = define_grid(len(numeric))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    plt.suptitle('Single Numeric Variables Boxplots', fontsize=11)

    for n in range(len(numeric)):
        axs[i, j].set_title("Boxplot for %s" % numeric[n])
        axs[i, j].boxplot(data[numeric[n]].dropna().values)
        axs[i, j].set_xticklabels([numeric[n]])  # Set the label to the variable name
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.savefig(f"dataset_health/profiling/distribution/images/covid_single_boxplots.png")
    show()
else:
    print("There are no numeric variables.")


# Outliers study 
if [] != numeric:
    outliers: dict[str, int] = count_outliers(data, numeric)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"dataset_health/profiling/distribution/images/covid_outliers_standart.png")
    show()
else:
    print("There are no numeric variables.")


target = "CovidPos"

values: Series = data[target].value_counts()
print(values)

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"dataset_health/profiling/distribution/images/covid_class_dist.png")
show()


# Numeric Histograms
if [] != numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i: int
    j: int
    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data[numeric[n]].dropna().values, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"dataset_health/profiling/distribution/images/covid_histograms_numeric.png")
    show()
else:
    print("There are no numeric variables.")


# Other Histograms
max_label_length = 15  # Define the maximum length for x-axis labels
symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
if [] != symbolic:
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0

    plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for the space between rows

    for n in range(len(symbolic)):
        counts: Series = data[symbolic[n]].value_counts()
        
        # Truncate labels longer than max_label_length
        truncated_labels = [
            label[:max_label_length] + '...' + label[-4:] if len(label) > max_label_length else label
            for label in counts.index.to_list()
        ]
        
        plot_bar_chart(
            truncated_labels,
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % symbolic[n],
            xlabel=symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    savefig(f"dataset_health/profiling/distribution/images/covid_histograms_symbolic.png", bbox_inches='tight')
    show()
else:
    print("There are no symbolic variables.")

