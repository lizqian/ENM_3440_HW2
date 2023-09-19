# Quantifying the Relationship Between Brain Waves and Attention
**Abstract**
This project explores the correlation between brain waves measured by electroencephalography (EEG) data and attention scores. The specific casual question being asked is, "How does brain wave type affect the ability to pay attention?". Using a dataset containing EEG measurements and corresponding attention scores, a linear regression model was developed to establish a predictive relationship between brain wave frequencies and attention. The model was trained and evaluated, achieving a reasonable level of predictive accuracy as indicated by a low Mean Squared Error (MSE) and a high R-squared (R2) value. The project includes visualizations such as scatterplots with lines of best fit, enabling the assessment of how well the model's predictions align with actual attention scores. Additionally, a diagonal line indicates a 1:1 relationship between predicted and actual scores. This research has implications for applications in neuroscience, cognitive science, and healthcare, as it provides a method for estimating attention levels non-invasively through EEG data. The model's predictive capabilities could be leveraged in various domains, including educational technology, cognitive assessment, and mental health diagnostics. Further research may involve refining the model and expanding its applicability across diverse datasets and populations.

**Confounders**
Each person's brain varies on the molecular and anatomical level, leading to many cofounders that may influence data pertaining to brain waves. Some cofounders include age, preexisting medical conditions, education levels, and caffeine or other substance use that may effect. Unfortunately, I do not have any of these cofounders measured in my dataset. Normalizing is not realistic option, and without having data on any of the cofounders, the only way to address them is by using a large enough dataset that these cofounders do not have a large affect on the overall results. 

**Colliders**
Colliders include the subject's movements or decissions during this time period, stress levels, and external stimuli, because all these can change both the brain waves being produced, as well as whether a person stays attentive. These colliders are not important in this case because this data was collected with an EEG machine, which means the subjects were almost certainly all kept in similar environments, not allowed to move a lot, and had limited stimuli. 

**Data Source, Cleaning, and Preprocessing**
This data set is a public dataset on Kaggle, and is added as a file to this repository. Since the data was already well-maintained on Kaggle, there was not much to do in terms of cleaning and preparing it other than checking for missing values. 

```
import pandas as pd
csv_file_path = 'acquiredDataset.csv'
df = pd.read_csv(csv_file_path)
missing_values = df.isnull().sum()
df_cleaned = df.dropna()
print("Missing Values:")
print(missing_values)
print("\nCleaned DataFrame:")
print(df_cleaned.head())
```

**Exploratory Data Analysis**
I plotted each type of brain wave I was interested in as a histogram to get a better sense of the distribution of data over each type of brain wave. 

```
import pandas as pd
import matplotlib.pyplot as plt
csv_file_path = 'acquiredDataset.csv'
df = pd.read_csv(csv_file_path)
def plot_histogram(data, title, x_label):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=20, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
plot_histogram(df['delta'], 'Histogram of Delta Brain Waves', 'Delta')
plot_histogram(df['theta'], 'Histogram of Theta Brain Waves', 'Theta')
plot_histogram(df['highAlpha'], 'Histogram of High Alpha Brain Waves', 'High Alpha')
plot_histogram(df['highBeta'], 'Histogram of High Beta Brain Waves', 'High Beta')
plot_histogram(df['highGamma'], 'Histogram of High Gamma Brain Waves', 'High Gamma')
```
/Users/elizabethqian/ENM_3440/output.png


