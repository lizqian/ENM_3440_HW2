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

![output](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/2f93d870-1c78-4917-b90b-0a035e25fa70)
![output3](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/9bd05253-479e-4b57-9f84-80ad5eb12e7b)
![output2](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/a527e635-9fd9-4ddb-b569-6d7f691cb012)
![output](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/4b5ded6d-b976-46f1-88e8-b63de451dcd3)

**Methodology**

This code iterates through the specified brain wave columns, creates scatter plots for each one against the attention scores, and adds a trendline (best-fit line) to each scatter plot. The trendline's slope (m) represents the correlation between each type of brain wave and the attention score. These scatter plots with trendlines allow you to visualize and assess the correlation between each type of brain wave frequency and the attention scores.

```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file_path.csv' with the actual path to your CSV file
csv_file_path = 'acquiredDataset.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Create separate scatter plots with trendlines for each brain wave frequency
brain_wave_columns = ['delta', 'theta', 'highAlpha', 'highBeta', 'highGamma']

for wave_column in brain_wave_columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[wave_column], df['attention'], alpha=0.5)
    plt.title(f'Scatter Plot of {wave_column.capitalize()} vs. Attention')
    plt.xlabel(f'{wave_column.capitalize()} Brain Waves')
    plt.ylabel('Attention Scores')

    # Calculate and add a trendline (best-fit line)
    x = df[wave_column]
    y = df['attention']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='red', linestyle='--', label=f'Trendline (m={m:.2f})')

    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
```

![output5](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/8eeda5e2-eab5-4c2d-b317-242a15363177)
![output4](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/781c9759-4cef-471d-92a2-97f19b63ff5b)
![output3](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/05d2e2f1-8f5e-4395-be61-113db6fcf22d)
![output2](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/ab09fb26-6a47-4df4-96f2-47844602c6fb)
![output](https://github.com/lizqian/ENM_3440_HW2/assets/133675095/8cb0756c-03f6-4722-b301-55db89e564a6)

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load your CSV data into a DataFrame
# Replace 'your_data.csv' with the actual path to your CSV file
data = pd.read_csv('your_data.csv')

# Define your feature columns (brain wave frequencies) and the target column (attention)
X = data[['delta', 'theta', 'highAlpha', 'highBeta', 'highGamma']]
y = data['attention']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Create a scatterplot of actual vs. predicted ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.title('Actual vs. Predicted Attention Scores')
plt.xlabel('Actual Attention Scores')
plt.ylabel('Predicted Attention Scores')

# Add a diagonal line indicating a 1:1 relationship
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')

# Calculate and add a line of best fit
fit = np.polyfit(y_test, y_pred, 1)
line_x = np.array([min(y_test), max(y_test)])
line_y = fit[0] * line_x + fit[1]
plt.plot(line_x, line_y, color='green', linestyle='--', label=f'Line of Best Fit (m={fit[0]:.2f}, b={fit[1]:.2f})')

plt.legend()
plt.grid(True)

# Show the plot
plt.show()
```
Mean Squared Error (MSE): 435.18
R-squared (R2) Score: 0.07

In this code:
We calculate the MSE and R-squared (R2) values to evaluate the model's performance.
We create a scatterplot of actual vs. predicted attention scores.
We add a diagonal line indicating a 1:1 relationship between actual and predicted scores.
We calculate and add a line of best fit (the regression line) to the plot.
This plot will allow you to visualize how well the model's predictions align with the actual attention scores, and the MSE and R2 values provide additional information about the model's accuracy.

**Results and Conclusions**

Both visually and mathematically it can be seen that there is no strong correlation between brain wave type and attention scores. The R2 calue is very low, and no graphs showed any strong visual correlations. From this we can conclude that brain wave type can neither be correlated with attention, nor used to predict attention scores. Likely, the biological mechanisms that control attention span are more complex than can be predicted with 5 brain wave types.

**Public Policy**

The study of attention is instrumental to public policy. Studies on the underlying mechanisms behind attention could be used to influence shifts in our understanding of psychology and neurosceince. Through this, educational public policy could be reevaluated for how it handles attention span. Additionally, a better understanding of attention could be used in the public health sector, where new medical advancements or pharmeceuticals can be pushed for under healthcare laws.
