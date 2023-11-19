# %%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart
dataset = pd.read_csv('class_credit_score.csv', sep=',', decimal='.', na_values='')
dataset.head(8)

# %%
ids = dataset['ID'].map(lambda x: int(x, 16)).sort_values()

# Initialize variables to count gaps and identify missing IDs
total_gaps = 0
missing_ids = []
gap_lengths = []

# Iterate through the ID list to check for gaps
for i in range(len(ids) - 1):
    current_id = ids[i]  # Convert hexadecimal ID to integer
    next_id = ids[i + 1]  # Convert next ID to integer
    
    # Check if the next ID is not consecutive
    if next_id != current_id + 1:
        total_gaps += 1
        missing_ids.extend(range(current_id + 1, next_id))  # Store missing IDs
        gap_lengths.append(next_id - current_id - 1)

missing = pd.Series(missing_ids).map(lambda x: hex(x)).tolist()

# Print the total number of gaps and missing IDs
print(f"Total number of gaps in IDs: {total_gaps}")
print(f"Missing IDs: {missing}")
print(f"Number of missing IDs: {len(missing)}")
print(f"Number of gaps with size 4: {pd.Series(gap_lengths)[pd.Series(gap_lengths) == 4].count()}")
print(f"Size of gaps in average: {len(missing) / total_gaps}")

figure(figsize=(8, 4))
plot_bar_chart(
    list(["Num Records", "Num Missing Records", "Num Gaps", "Num Gaps Size 4"]),
    list([len(ids), len(missing), total_gaps, pd.Series(gap_lengths)[pd.Series(gap_lengths) == 4].count()]),
    title="Granularity of IDs"
)
savefig(f"images/figure_gran_1.png")
show()

# %% [markdown]
# Através desta informação pode-se assumir que este dataset tem em falta informações relativas aos meses de Setembro, Outubro, Novembro e Dezembro.

# %% [markdown]
# ## Customer_ID

# %%
# Determinar IDs mal formatados:
n_wrong_format = dataset.loc[dataset['Customer_ID'].str[:4] == "CUS_", 'Customer_ID'].count()
print(f"Number of correctly formatted customer IDs: {n_wrong_format}")

# %%
ids_repeat = dataset["Customer_ID"].map(lambda x: x[4:]).map(lambda x: int(x, 16))
ids = pd.Series(ids_repeat.unique()).sort_values()

total_gaps = 0
missing_ids = []
gap_lengths = []

# Iterate through the ID list to check for gaps
for i in range(len(ids) - 1):
    current_id = ids[i]  # Convert hexadecimal ID to integer
    next_id = ids[i + 1]  # Convert next ID to integer
    
    # Check if the next ID is not consecutive
    if next_id != current_id + 1:
        total_gaps += 1
        gap_lengths.append(next_id - current_id - 1)

# Print the total number of gaps and missing IDs
print(f"Total number of gaps in IDs: {total_gaps}")
print(f"Missing IDs: {missing_ids}")
print(f"Number of missing IDs: {len(missing_ids)}")
print(f"Number of gaps with size 4: {pd.Series(gap_lengths)[pd.Series(gap_lengths) == 4].count()}")
print(f"Size of gaps in average: {len( missing_ids) / total_gaps}")

figure(figsize=(8, 4))
plot_bar_chart(
    list(["Num IDs", "Num Missing IDs", "Num Gaps", "Num Gaps Size 4"]),
    list([len(ids), len(missing_ids), total_gaps, pd.Series(gap_lengths)[pd.Series(gap_lengths) == 4].count()]),
    title="Granularity of Customer IDs"
)
savefig(f"images/figure_gran_2.png")
show()


## IGNORE

# %%
ids_repeat = dataset["Customer_ID"].map(lambda x: x[4:]).map(lambda x: int(x, 16))
ids = pd.Series(ids_repeat.unique()).sort_values()

print(ids)

# %% [markdown]
# ## Month
# 
# Separar por mês

# %%
mv = {}
for month in dataset["Month"].unique():
    mv[month] = dataset.loc[dataset["Month"] == month, "Month"].count()

figure(figsize=(8, 4))
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    xlabel="Month",
    ylabel="Ocurrences",
    title="Month Frequency"
)
savefig(f"images/figure_gran_month.png")
show()

# %% [markdown]
# ## Name
# Categorizar por detalhe de nomes

# %%
names = dataset["Name"]
name_lengths = names.map(lambda x: str(x).replace(" ", "")).map(lambda x: len(x))
n_words = names.map(lambda x: str(x).count(" "))

n_word_freq, name_lengths_freq = {}, {}
min_word_freq = name_lengths.min()
max_word_freq = name_lengths.max()

for i in range(min_word_freq, max_word_freq + 1):
    name_lengths_freq[i] = name_lengths.loc[name_lengths == i].count()

min_n_words = n_words.min()
max_n_words = n_words.max()

for i in range(min_n_words, max_n_words + 1):
    n_word_freq[i+1] = n_words.loc[n_words == i].count()

print(n_word_freq, name_lengths_freq)


figure(figsize=(8, 4))
plot_bar_chart(
    list(name_lengths_freq.keys()),
    list(name_lengths_freq.values()),
    xlabel="Name Length in characters",
    ylabel="Ocurrences",
    title="Name Length Granularity"
)
savefig(f"images/figure_gran_name_1.png")
show()

figure(figsize=(6, 4))
plot_bar_chart(
    list(n_word_freq.keys()),
    list(n_word_freq.values()),
    xlabel="Number of words in names",
    ylabel="Ocurrences",
    title="Name Composition Granularity"
)
savefig(f"images/figure_gran_name_2.png")
show()


# %% [markdown]
# A partir destes gráficos podemos ver que existe uma certa falta de detalhe nos nomes dos clientes visto que na maioria apenas temos um dos seus nomes. 

# %% [markdown]
# ## Age
# Começar sem intervalos e ajustar

# %%
d = dataset["Age"].value_counts().sort_index()
figure(figsize=(25, 4))
plot_bar_chart(
    list(d.index),
    list(d.values),
    xlabel="Age",
    ylabel="Ocurrences",
    title="Age Granularity"
)
savefig(f"images/figure_gran_age_with_outliers.png")
show()

# %%
numeric_ages = pd.to_numeric(dataset['Age'], errors='coerce')
valid_ages = numeric_ages[~numeric_ages.isnull()]

Q1 = valid_ages.quantile(0.25)
Q3 = valid_ages.quantile(0.75)
IQR = Q3 - Q1

# Calculate lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out outliers
filtered_ages = valid_ages[(valid_ages >= lower_bound) & (valid_ages <= upper_bound)]
d = filtered_ages.value_counts().sort_index()

figure(figsize=(25, 4))
plot_bar_chart(
    list(d.index),
    list(d.values),
    xlabel="Age",
    title="Age Granularity"
)
savefig(f"images/figure_gran_age_without_outliers_all_bins.png")
show()

# %%
numeric_ages = pd.to_numeric(dataset['Age'], errors='coerce')
valid_ages = numeric_ages[~numeric_ages.isnull()]

Q1 = valid_ages.quantile(0.25)
Q3 = valid_ages.quantile(0.75)
IQR = Q3 - Q1

# Calculate lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out outliers
filtered_ages = valid_ages[(valid_ages >= lower_bound) & (valid_ages <= upper_bound)]

# Binning ages into three categories
bins = [0, 18, 25, 41, 57, float('inf')]
labels = ['< 18', '18-24', '25-40', '41-56', '> 56']
dataset['Age Category'] = pd.cut(filtered_ages, bins=bins, labels=labels, right=False)

age_category_counts = dataset['Age Category'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(14, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Age",
    title="Age Granularity without Outliers"
)

savefig(f"images/figure_gran_age_without_outliers_binned.png")
show()

# %% [markdown]
# ## SSN
# AAA-GG-SSSS
# area - group - serial

# %% [markdown]
# ## Occupation

# %%
occupations = dataset["Occupation"]

d = occupations.value_counts().sort_index()

figure(figsize=(8, 4))
plot_bar_chart(
    d.index,
    d.values,
    xlabel="Occupations",
    ylabel="Ocurrences",
    title="Occupation Granularity"
)

savefig(f"images/figure_gran_occu_all_bins.png")
show()

# %%
occupations = dataset["Occupation"]

job_to_field = {
    'Doctor': 'Medical and Healthcare',
    'Surgeon': 'Medical and Healthcare',
    'Engineer': 'Engineering and Technology',
    'Entrepreneur': 'Business and Entrepreneurship',
    'Manager': 'Business and Entrepreneurship',
    'Teacher': 'Education',
    'Developer': 'Information Technology',
    'Media_Manager': 'Information Technology',
    'Lawyer': 'Legal',
    'Journalist': 'Media and Journalism',
    'Accountant': 'Finance and Accounting',
    'Musician': 'Creative Arts',
    'Writer': 'Creative Arts',
    'Architect': 'Architecture and Design',
    'Mechanic': 'Engineering and Technology'
}

dataset['General_Field'] = dataset['Occupation'].map(job_to_field)

d = dataset['General_Field'].value_counts().sort_index()

figure(figsize=(8, 4))
plot_bar_chart(
    d.index,
    d.values,
    xlabel="Occupations",
    ylabel="Ocurrences",
    title="Occupation Granularity"
)

savefig(f"images/figure_gran_occu_generalized.png")
show()

# %% [markdown]
# ## Annual Income

# %%
print(dataset["Annual_Income"].min())
print(dataset["Annual_Income"].max())
print(dataset["Annual_Income"].median())
print(pd.to_numeric(dataset['Annual_Income'], errors='coerce').quantile(0.25))
print(pd.to_numeric(dataset['Annual_Income'], errors='coerce').quantile(0.75))

# %%
# Binning ages into three categories
bins = [0, 20000, 38000, 73000, float('inf')]
labels = ['< 20', '20-38', '38-73', '> 73']
dataset['Age Category'] = pd.cut(dataset["Annual_Income"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['Age Category'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(14, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Annual Income (thousands)",
    title="Annual Income Granularity"
)

savefig(f"images/figure_gran_an_inc_with_outliers_1.png")
show()

# %%
# Binning ages into three categories
bins = [0, 20000, 30000, 40000, 50000, float('inf')]
labels = ['< 20', '20-30', '30-40', '40-50', '> 50']
dataset['Age Category'] = pd.cut(dataset["Annual_Income"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['Age Category'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(14, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Annual Income (thousands)",
    title="Annual Income Granularity"
)

savefig(f"images/figure_gran_an_inc_with_outliers_2.png")
show()

# %% [markdown]
# ## Number of Bank Accounts

# %%
print(dataset["Num_Bank_Accounts"].min())
print(dataset["Num_Bank_Accounts"].max())
print(dataset["Num_Bank_Accounts"].median())
print(pd.to_numeric(dataset['Num_Bank_Accounts'], errors='coerce').quantile(0.25))
print(pd.to_numeric(dataset['Num_Bank_Accounts'], errors='coerce').quantile(0.75))

# %%
coerced = pd.to_numeric(dataset['Num_Bank_Accounts'], errors='coerce')
valid = coerced[~coerced.isnull()]

Q1 = valid.quantile(0.25)
Q3 = valid.quantile(0.75)
IQR = Q3 - Q1

# Calculate lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out outliers
filtered = valid[(valid >= lower_bound) & (valid <= upper_bound)]

final = filtered.value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(16, 4))
plot_bar_chart(
    final.index,
    final.values,
    xlabel="Number of bank accounts",
    title="Number of bank accounts Frequency"
)

savefig(f"images/figure_gran_n_banks_1.png")
show()

# %%
# Binning ages into three categories
bins = [-1, 3, 5, 8, float('inf')]
labels = ['< 3', '3-5', '6-8', '> 8']
dataset['NBanks Category'] = pd.cut(dataset["Num_Bank_Accounts"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['NBanks Category'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(8, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Number of bank accounts",
    title="Number of bank accounts Frequency"
)

savefig(f"images/figure_gran_n_banks_2.png")
show()

# %% [markdown]
# ## Num_credit_card

# %%
coerced = pd.to_numeric(dataset["Num_Credit_Card"], errors='coerce')
valid = coerced[~coerced.isnull()]

Q1 = valid.quantile(0.25)
Q3 = valid.quantile(0.75)
IQR = Q3 - Q1

# Calculate lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out outliers
filtered = valid[(valid >= lower_bound) & (valid <= upper_bound)]

final = filtered.value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(16, 4))
plot_bar_chart(
    final.index,
    final.values,
    xlabel="Number of credit cards",
    title="Number of credit cards Frequency"
)

savefig(f"images/figure_gran_n_cards_1.png")
show()

# %%
# Binning ages into three categories
bins = [-1, 3, 5, 8, float('inf')]
labels = ['< 3', '3-5', '6-8', '> 8']
dataset['NBanks Category'] = pd.cut(dataset["Num_Credit_Card"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['NBanks Category'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(8, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Number of credit cards",
    title="Number of credit cards Frequency"
)

savefig(f"images/figure_gran_n_cards_2.png")
show()

# %% [markdown]
# ## Interest rate

# %%
print(dataset["Interest_Rate"].min())
print(dataset["Interest_Rate"].max())
print(dataset["Interest_Rate"].median())
print(pd.to_numeric(dataset['Interest_Rate'], errors='coerce').quantile(0.25))
print(pd.to_numeric(dataset['Interest_Rate'], errors='coerce').quantile(0.75))

# %%


# Binning ages into three categories
bins = [1, 5, 10, 15, float('inf')]
labels = ['< 5', '5-10', '10-15', '> 15']
dataset['new'] = pd.cut(dataset["Interest_Rate"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['new'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(8, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Interest rate %",
    title="Interest rate frequency"
)

savefig(f"images/figure_gran_int_rate_1.png")
show()

# %%
# Binning ages into three categories
bins = [1, 8, 13, 20, float('inf')]
labels = ['< 8', '8-13', '13-20', '> 20']
dataset['new'] = pd.cut(dataset["Interest_Rate"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['new'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(8, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Interest rate %",
    title="Interest rate frequency"
)

savefig(f"images/figure_gran_int_rate_2.png")
show()

# %% [markdown]
# ## Num of Loan

# %%
print(dataset["NumofLoan"].min())
print(dataset["NumofLoan"].max())
print(dataset["NumofLoan"].median())
print(pd.to_numeric(dataset['NumofLoan'], errors='coerce').quantile(0.1))
print(pd.to_numeric(dataset['NumofLoan'], errors='coerce').quantile(0.25))
print(pd.to_numeric(dataset['NumofLoan'], errors='coerce').quantile(0.75))

# %%


# Binning ages into three categories
bins = [1, 3, 6, 10, float('inf')]
labels = ['< 3', '3-6', '6-10', '> 10']
dataset['new'] = pd.cut(dataset["Interest_Rate"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['new'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(8, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Number of loans",
    title="Number of loans frequency"
)

savefig(f"images/figure_gran_loan_1.png")
show()

# %%
# Binning ages into three categories
bins = [1, 7, 12, 20, 35, float('inf')]
labels = ['1-7', '7-12', '12-20', '20-35', '> 35']
dataset['new'] = pd.cut(dataset["Interest_Rate"], bins=bins, labels=labels, right=False)

age_category_counts = dataset['new'].value_counts().sort_index()

# Create a bar plot without outliers
figure(figsize=(8, 4))
plot_bar_chart(
    age_category_counts.index,
    age_category_counts.values,
    xlabel="Number of loans",
    title="Number of loans frequency"
)

savefig(f"images/figure_gran_loan_1.png")
show()

# %% [markdown]
# ## Final figure - All numerical

# %%
import matplotlib.pyplot as plt
import dslabs_functions as ds
import re

units = {
    "Age": "(Years)",
    "Annual_Income": "($)",
	"Monthly_Inhand_Salary": "($)",
	"Num_Bank_Accounts": "",
    "Num_Credit_Card": "",
	"Interest_Rate": "(%)",
	"NumofLoan": "",
	"Delay_from_due_date": "(days)",
    "NumofDelayedPayment": "",
	"ChangedCreditLimit": "",
	"NumCreditInquiries": "",
    "OutstandingDebt": "($)",
	"CreditUtilizationRatio": "",
	"TotalEMIpermonth": "($)",
    "Amountinvestedmonthly": "($)",
	"MonthlyBalance": "($)",
    "CreditHistoryAge": "(Years)"
}
print(len(units))

hist_age = dataset["Credit_History_Age"]
hist_vals = []

for val in hist_age:
    matches = re.findall(r'(\d+) Years and (\d+) Months', str(val))
    if matches == []:
        continue
    else:
        hist_vals += [float(matches[0][0]) + ((float(matches[0][1])-1)/12) ,]

dataset["CreditHistoryAge"] = pd.Series(hist_vals).astype(float)

t = dataset.select_dtypes(include=['number', 'float'])
t["Age"] = pd.to_numeric(dataset['Age'], errors='coerce')
variables = t.columns
rows, cols = ds.define_grid(len(variables))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s' % variables[n])
    axs[i, j].set_xlabel(f'{variables[n]} {units[variables[n]]}')
    axs[i, j].set_ylabel('Number of Records')

    # Adjusting the rwidth parameter to increase the distance between bars along the x-axis
    axs[i, j].hist(t[variables[n]].values, bins=5, color='lightblue', rwidth=0.9)  # Adjust rwidth

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

savefig(f"images/figure_granularity_5.png")  # Save the figure
show()

# %%
import matplotlib.pyplot as plt
import dslabs_functions as ds
import re

units = {
    "Age": "(Years)",
    "Annual_Income": "($)",
	"Monthly_Inhand_Salary": "($)",
	"Num_Bank_Accounts": "",
    "Num_Credit_Card": "",
	"Interest_Rate": "(%)",
	"NumofLoan": "",
	"Delay_from_due_date": "(days)",
    "NumofDelayedPayment": "",
	"ChangedCreditLimit": "",
	"NumCreditInquiries": "",
    "OutstandingDebt": "($)",
	"CreditUtilizationRatio": "",
	"TotalEMIpermonth": "($)",
    "Amountinvestedmonthly": "($)",
	"MonthlyBalance": "($)",
    "CreditHistoryAge": "(Years)"
}
print(len(units))

hist_age = dataset["Credit_History_Age"]
hist_vals = []

for val in hist_age:
    matches = re.findall(r'(\d+) Years and (\d+) Months', str(val))
    if matches == []:
        continue
    else:
        hist_vals += [float(matches[0][0]) + ((float(matches[0][1])-1)/12) ,]

dataset["CreditHistoryAge"] = pd.Series(hist_vals).astype(float)

t = dataset.select_dtypes(include=['number', 'float'])
t["Age"] = pd.to_numeric(dataset['Age'], errors='coerce')
variables = t.columns
rows, cols = ds.define_grid(len(variables))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s' % variables[n])
    axs[i, j].set_xlabel(f'{variables[n]} {units[variables[n]]}')
    axs[i, j].set_ylabel('Number of Records')

    # Adjusting the rwidth parameter to increase the distance between bars along the x-axis
    axs[i, j].hist(t[variables[n]].values, bins=10, color='lightblue', rwidth=0.9)  # Adjust rwidth

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

savefig(f"images/figure_granularity_10.png")  # Save the figure
show()

# %%
import matplotlib.pyplot as plt
import dslabs_functions as ds
import re

units = {
    "Age": "(Years)",
    "Annual_Income": "($)",
	"Monthly_Inhand_Salary": "($)",
	"Num_Bank_Accounts": "",
    "Num_Credit_Card": "",
	"Interest_Rate": "(%)",
	"NumofLoan": "",
	"Delay_from_due_date": "(days)",
    "NumofDelayedPayment": "",
	"ChangedCreditLimit": "",
	"NumCreditInquiries": "",
    "OutstandingDebt": "($)",
	"CreditUtilizationRatio": "",
	"TotalEMIpermonth": "($)",
    "Amountinvestedmonthly": "($)",
	"MonthlyBalance": "($)",
    "CreditHistoryAge": "(Years)"
}
print(len(units))

hist_age = dataset["Credit_History_Age"]
hist_vals = []

for val in hist_age:
    matches = re.findall(r'(\d+) Years and (\d+) Months', str(val))
    if matches == []:
        continue
    else:
        hist_vals += [float(matches[0][0]) + ((float(matches[0][1])-1)/12) ,]

dataset["CreditHistoryAge"] = pd.Series(hist_vals).astype(float)

t = dataset.select_dtypes(include=['number', 'float'])
t["Age"] = pd.to_numeric(dataset['Age'], errors='coerce')
variables = t.columns
rows, cols = ds.define_grid(len(variables))
fig, axs = plt.subplots(rows, cols, figsize=(cols*ds.HEIGHT, rows*ds.HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s' % variables[n])
    axs[i, j].set_xlabel(f'{variables[n]} {units[variables[n]]}')
    axs[i, j].set_ylabel('Number of Records')

    # Adjusting the rwidth parameter to increase the distance between bars along the x-axis
    axs[i, j].hist(t[variables[n]].values, bins=15, color='lightblue', rwidth=0.9)  # Adjust rwidth

    i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

savefig(f"images/figure_granularity_15.png")  # Save the figure
show()

# %% [markdown]
# ## Final figure - All symbolic

# %%
# Select columns with numerical and string data types
string_cols = dataset.select_dtypes(include="object").columns
symbolic_cols = [col for col in string_cols if col not in ["ID", "Name", "Customer_ID", "SSN", "Age", "Credit_History_Age"]]

# Combine numerical and string columns
rows, cols = ds.define_grid(len(symbolic_cols))

print(cols * ds.HEIGHT, rows * ds.HEIGHT)
fig, axs = plt.subplots(rows, cols, figsize=(cols * ds.HEIGHT, rows * ds.HEIGHT))
i, j = 0, 0
plt.subplots_adjust(hspace=1)

            
types = dataset["Type_of_Loan"]

types_values = []
for text in types:
    matches = re.findall(r'([\w\s-]+)(?:, | and |$)', str(text))
    m_filtered = [m if "and" not in m else m[4:] for m in matches]
    types_values += m_filtered

final = pd.Series(types_values).value_counts()

axs[i, j].set_title('Histogram for %s' % "Type_of_Loan")
axs[i, j].set_xlabel("Type_of_Loan")
axs[i, j].set_ylabel('Number of Records')
axs[i, j].bar(final.index, final.values, color = "lightblue")
axs[i, j].set_xticklabels(final.index, rotation=90)
i, j = i, j+1

for col in symbolic_cols:
    if col != "Type_of_Loan":
        axs[i, j].set_title('Histogram for %s' % col)
        axs[i, j].set_xlabel(col)
        axs[i, j].set_ylabel('Number of Records')
        value_counts = dataset[col].value_counts()
        axs[i, j].bar(value_counts.index, value_counts.values, color='lightblue')
        axs[i, j].set_xticklabels(value_counts.index, rotation=90)
        i, j = (i + 1, 0) if (j + 1) % cols == 0 else (i, j + 1)
    
    


plt.savefig(f"images/figure_granularity_symbolic.png")  # Save the figure
plt.show()

# %% [markdown]
# # Sparsity

# %%
import pandas as pd
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart

vars = pd.DataFrame()
for col in dataset.select_dtypes(include="number").columns:
    vars[col] = pd.to_numeric(dataset[col], errors='coerce')
symbolic_vars = dataset.select_dtypes(include="object").columns
vars_symbolic = pd.DataFrame()

for col in symbolic_vars:
    vars_symbolic[col], _ = pd.factorize(dataset[col])

# Combine numeric and factorized symbolic variables
vars_combined = pd.concat([vars, vars_symbolic], axis=1)

n = len(vars_combined.columns)
fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)

for i in range(len(vars_combined.columns)):
    var1 = vars_combined.columns[i]
    for j in range(i + 1, len(vars_combined.columns)):
        var2 = vars_combined.columns[j]
        plot_multi_scatters_chart(vars_combined, var1, var2, ax=axs[i, j - 1])

savefig(f"images/sparsity_per_class_study.png")
show()

# %% [markdown]
# # Correlation

# %%
from seaborn import heatmap
from dslabs_functions import get_variable_types
import re

exclude = ["ID", "Customer_ID", "Name", "SSN", "Age", "Credit_History_Age"]

# Create a copy of the dataset
modified_dataset = dataset.copy()

hist_age = modified_dataset["Credit_History_Age"]
hist_vals = []

for val in hist_age:
    matches = re.findall(r'(\d+) Years and (\d+) Months', str(val))
    if matches == []:
        continue
    else:
        hist_vals.append(float(matches[0][0]) + ((float(matches[0][1]) - 1) / 12))

# Add a new column 'CreditHistoryAge' to the modified dataset
modified_dataset["CreditHistoryAge"] = pd.Series(hist_vals).astype(float)

# Select numeric columns
numeric = modified_dataset.select_dtypes(include=['number', 'float'])

# Convert the 'Age' column to numeric without altering the original dataset
numeric["Age"] = pd.to_numeric(modified_dataset['Age'], errors='coerce')

symbolic_vars = [col for col in modified_dataset.select_dtypes(include="object") if col not in exclude]
symbolic_factors = pd.DataFrame()

for col in symbolic_vars:
    symbolic_factors[col], _ = pd.factorize(modified_dataset[col])

# Combine the factorized symbolic variables with the numeric variables
all_vars = pd.concat([numeric, symbolic_factors], axis=1)

# Compute the correlation matrix for all variables
corr_mtx = all_vars.corr().abs()

# Generate the heatmap
figure()
heatmap(
    abs(corr_mtx),
    xticklabels=all_vars.columns,
    yticklabels=all_vars.columns,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/correlation_analysis_all.png")
show()

# %%
from seaborn import heatmap
from dslabs_functions import get_variable_types
import re

exclude = ["ID", "Customer_ID", "Name", "SSN", "Age", "Credit_History_Age"]

# Create a copy of the dataset
modified_dataset = dataset.copy()

hist_age = modified_dataset["Credit_History_Age"]
hist_vals = []

for val in hist_age:
    matches = re.findall(r'(\d+) Years and (\d+) Months', str(val))
    if matches == []:
        continue
    else:
        hist_vals.append(float(matches[0][0]) + ((float(matches[0][1]) - 1) / 12))

# Add a new column 'CreditHistoryAge' to the modified dataset
modified_dataset["CreditHistoryAge"] = pd.Series(hist_vals).astype(float)

# Select numeric columns
numeric = modified_dataset.select_dtypes(include=['number', 'float'])

# Convert the 'Age' column to numeric without altering the original dataset
numeric["Age"] = pd.to_numeric(modified_dataset['Age'], errors='coerce')

# Compute the correlation matrix for all variables
corr_mtx = numeric.corr().abs()

# Generate the heatmap
figure()
heatmap(
    abs(corr_mtx),
    xticklabels=numeric.columns,
    yticklabels=numeric.columns,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/correlation_analysis_numerical.png")
show()

# %%
from seaborn import heatmap
from dslabs_functions import get_variable_types
import re

exclude = ["ID", "Customer_ID", "Name", "SSN", "Age", "Credit_History_Age"]

# Create a copy of the dataset
modified_dataset = dataset.copy()

hist_age = modified_dataset["Credit_History_Age"]
hist_vals = []

for val in hist_age:
    matches = re.findall(r'(\d+) Years and (\d+) Months', str(val))
    if matches == []:
        continue
    else:
        hist_vals.append(float(matches[0][0]) + ((float(matches[0][1]) - 1) / 12))

# Add a new column 'CreditHistoryAge' to the modified dataset
modified_dataset["CreditHistoryAge"] = pd.Series(hist_vals).astype(float)

# Select numeric columns
numeric = modified_dataset.select_dtypes(include=['number', 'float'])

# Convert the 'Age' column to numeric without altering the original dataset
numeric["Age"] = pd.to_numeric(modified_dataset['Age'], errors='coerce')

symbolic_vars = [col for col in modified_dataset.select_dtypes(include="object") if col not in exclude]
symbolic_factors = pd.DataFrame()

for col in symbolic_vars:
    symbolic_factors[col], _ = pd.factorize(modified_dataset[col])

# Compute the correlation matrix for all variables
corr_mtx = symbolic_factors.corr().abs()

# Generate the heatmap
figure()
heatmap(
    abs(corr_mtx),
    xticklabels=symbolic_factors.columns,
    yticklabels=symbolic_factors.columns,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
savefig(f"images/correlation_analysis_symbolic.png")
show()

