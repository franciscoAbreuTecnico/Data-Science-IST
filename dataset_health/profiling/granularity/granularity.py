import pandas as pd

def categorize_age(age):
    if age == "Age 18 to 24":
        return "Young Adult"
    if age == "Age 25 to 29":
        return "Young Adult"
    if age == "Age 30 to 34":
        return "Young Adult"
    if age == "Age 35 to 39":
        return "Adult"
    if age == "Age 40 to 44":
        return "Adult"
    if age == "Age 45 to 49":
        return "Adult"
    if age == "Age 50 to 54":
        return "Adult"
    if age == "Age 55 to 59":
        return "Older Adult"
    if age == "Age 60 to 64":
        return "Older Adult"
    if age == "Age 65 to 69":
        return "Older Adult"
    if age == "Age 70 to 74":
        return "Older Adult"
    if age == "Age 75 to 79":
        return "Older Adult"
    if age == "Age 80 or older":
        return "Elderly"
    
def region(state):
    south = ["Alabama", "Arkansas", "Delaware", "District of Columbia", "Florida", "Georgia", "Kentucky", "Louisiana",
                "Maryland", "Mississippi", "North Carolina", "Oklahoma", "South Carolina", "Tennessee", "Texas",
                "Virginia", "West Virginia"]
    midwest = ["Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri", "Nebraska", "North Dakota",
                "Ohio", "South Dakota", "Wisconsin"]
    northeast = ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New Jersey", "New York", "Pennsylvania",
                    "Rhode Island", "Vermont"]
    west = ["Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada", "New Mexico",
                "Oregon", "Utah", "Washington", "Wyoming"]
    if state in south:
        return "South"
    if state in midwest:
        return "Midwest"
    if state in northeast:
        return "Northeast"
    if state in west:
        return "West"


filename = "dataset_health/data/class_pos_covid.csv"

df = pd.read_csv(filename)

df['AgeGroup'] = df['AgeCategory'].apply(categorize_age)

df['Region'] = df['State'].apply(region)

df.to_csv("dataset_health/data/class_pos_covid_derived.csv", index=False)
