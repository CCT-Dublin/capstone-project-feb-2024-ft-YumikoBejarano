#!/usr/bin/env python
# coding: utf-8

# # <center> CCT College Dublin
# # <center> Assessment Cover Page
# 
# <br><br><br>
# 
# ### Module Title:
# Strategic Thinking
# 
# ### Assessment Title:
# CA 2 Capstone Report 
# 
# ### Student Details:
# - **Full Name:**&nbsp;&nbsp;&nbsp; Yumiko Maria Bejarano Azogue  
# - **Student Number:**&nbsp;&nbsp;&nbsp; 2024144  
# - **Date of Submission:**&nbsp;&nbsp;&nbsp;24 May 2024
#     
# 
# <br><br><br>
# 
# ---
#     
# ## GITHUB
# https://github.com/CCT-Dublin/capstone-project-feb-2024-ft-YumikoBejarano
# 
# #### Declaration 
# 
# ```
# By submitting this assessment, I confirm that I have read the CCT policy on Academic Misconduct and understand the implications of submitting work that is not my own or does not appropriately reference material taken from a third party or other source. I declare it to be my own work and that all material from third parties has been appropriately referenced. I further confirm that this work has not previously been submitted for assessment by myself or someone else in CCT College Dublin or any other higher education institution.
# 
# ```
# <br><br><br>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

# Import thi slibrary to suppress the warnings
import warnings
# The object 'warnings' is used to call the method 'filterwarnings' and ignore the warnings
warnings.filterwarnings('ignore')  


# In[2]:


#Importing the dataset
df = pd.read_csv("Glassdoor Gender Pay Gap.csv")


# ### Data Description:
# Begin by describing the dataset, including its size, structure, and variables. This involves examining the types of variables (numeric or categorical), their distributions, and any missing or erroneous values.

# In[3]:


# Displaying the top 5 rows
print(df.head())


# In[4]:


# Displaying the bottom 5 rows
print(df.tail())


# In[5]:


# Displaying the dataset size
print("Dataset Size:", df.shape)

# Dataset Size: (1000, 9)


# In[6]:


# Printing the data structure
print('Datatype in Each Column\n')
pd.DataFrame({'Datatype': df.dtypes, 'Unique Values': df.nunique()}).rename_axis("Column Name")



# In[7]:


# Pairplot to visualize relationships between numerical features, color-coded by gender
sns.pairplot(df, hue="Gender", palette={"Female": "pink", "Male": "skyblue"})
plt.show()


# In[8]:


# Displaying information about the dataframe
print("\nInformation about the dataframe:")
print(df.info())



# In[9]:


# Identifying categorical variables
categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

# Print the names of categorical columns
print("Categorical variables:", categorical_cols)

# # Identifying categorical variables
# categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

# # Print the names of categorical columns
# print("Categorical variables:", categorical_cols)


# In[10]:


# Plotting the distribution of categorical variables
sns.set_palette(["pink", "skyblue"])  # Set the palette for Female and Male
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

for ax, col in zip(axes.flatten(), categorical_cols):
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()


# In[11]:


# Summary statistics for numerical variables
print("Descriptive Statistics for Numerical Variables:\n")
print(df.describe().T)



# ### Summary Statistics:
# 
# Calculate descriptive statistics for numerical variables (e.g., mean, median, standard deviation) and frequency tables for categorical variables.
# 
# This provides an initial understanding of the data's central tendencies and variability.

# In[12]:


#  Statistical information of Dataframe
df.describe().T


# In[13]:


# Summary Statistics
# Print descriptive statistics for numerical variables
print("Descriptive Statistics for Numerical Variables:\n" )
df.describe()


# In[14]:


# Identify categorical variables
print("Frequency Tables for Categorical Variables:\n")
df.describe(include=object).T



# In[ ]:





# ### Pre-processing

# #### Handling Missing Values:
# Address any missing data by imputation or removal, depending on the extent and nature of the missingness.

# In[15]:


# Checking for missing values
print("Missing Values:\n", df.isnull().sum())



# ## Variable: Salary
# 
# The salary is calculated as sum of the base salary and the yearly bonus.

# In[16]:


# Calculating the salary by summing up the base pay and bonus
df['Salary'] = df['BasePay'] + df['Bonus']


# ### Statistics Analysis

# In[17]:


statistics.mean(df['Salary'])

# 100939.814


# In[18]:


# Calculating the mean salary for males and females
male_mean_salary = df[df['Gender'] == 'Male']['Salary'].mean()
female_mean_salary = df[df['Gender'] == 'Female']['Salary'].mean()
overall_mean_salary = df['Salary'].mean()


# In[19]:


print("Overall Mean Salary:", overall_mean_salary)
print("Male Mean Salary:", male_mean_salary)
print("Female Mean Salary:", female_mean_salary)



# In[20]:


# Plotting the histogram of salaries with mean values
plt.figure(figsize=(10, 6))
plt.hist(df['Salary'], bins=20, color='lightblue', alpha=0.7)

plt.axvline(x=male_mean_salary, color='blue', linestyle='--', label=f'Male Mean Salary: {male_mean_salary:.2f}')
plt.axvline(x=female_mean_salary, color='magenta', linestyle='--', label=f'Female Mean Salary: {female_mean_salary:.2f}')
plt.axvline(x=overall_mean_salary, color='green', linestyle='--', label=f'Overall Mean Salary: {overall_mean_salary:.2f}')

plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Distribution of Salaries with Mean Values')
plt.legend()

plt.show()


# ### Encoding Categorical Variables: 
# Convert categorical variables into a numerical format suitable for analysis, such as one-hot encoding or label encoding.

# In[21]:


# Encoding categorical variables
label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])
        
label_encoder


# In[22]:


df


# # Exploratory Data Analysis (EDA)
# 

# In[23]:


data = df.copy()


# In[24]:


data.describe().T

# count	mean	std	min	25%	50%	75%	max
# JobTitle	1000.0	4.481	2.869821	0.0	2.00	5.0	7.00	9.0
# Gender	1000.0	0.532	0.499225	0.0	0.00	1.0	1.00	1.0
# Age	1000.0	41.393	14.294856	18.0	29.00	41.0	54.25	65.0
# PerfEval	1000.0	3.037	1.423959	1.0	2.00	3.0	4.00	5.0
# Education	1000.0	1.491	1.099604	0.0	1.00	1.0	2.00	3.0
# Dept	1000.0	2.046	1.414880	0.0	1.00	2.0	3.00	4.0
# Seniority	1000.0	2.971	1.395029	1.0	2.00	3.0	4.00	5.0
# BasePay	1000.0	94472.653	25337.493272	34208.0	76850.25	93327.5	111558.00	179726.0
# Bonus	1000.0	6467.161	2004.377365	1703.0	4849.50	6507.0	8026.00	11293.0
# Salary	1000.0	100939.814	25156.599655	40828.0	83443.00	100047.0	117656.00	184010.0


# In[25]:


# Correlation Matrix
c= data.corr()
c


# In[26]:


plt.figure(figsize=(8,6))
sns.heatmap(c,cmap="BrBG",annot=True)


# In[27]:


from sklearn.preprocessing import MinMaxScaler

# Standardize the numerical features
numerical_features = data.select_dtypes(include=[np.number])
numerical_features_list = numerical_features.columns.tolist()
numerical_features_list.remove('Gender')  # Remove 'Gender' as it is the dependent variable

numerical_features_list


# In[28]:


# Calculate the Interquartile Range (IQR) for each numerical feature
Q1 = numerical_features.quantile(0.25)
Q3 = numerical_features.quantile(0.75)
IQR = Q3 - Q1

# Define a criterion to identify outliers (e.g., values that are above or below 1.5 times the IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (numerical_features < lower_bound) | (numerical_features > upper_bound)
outliers_count = outliers.sum()

# Show the total number of outliers found for each column
print("Total outliers in each column:")
outliers_count



# In[29]:


# Remove outliers by keeping data below a specified percentile
percentage_to_keep = 0.95
filtered_data_frames = []

# Iterate over all columns except 'Gender'
for column in df.columns:
    if column != 'Gender':
        # Calculate the percentile value for the current column
        outlier = data[column].quantile(percentage_to_keep)
        # Filter the data keeping only those below the percentile
        filtered_data_frames.append(data[data[column] < outlier])

# Concatenate all filtered DataFrames into one
filtered_data_df2 = pd.concat(filtered_data_frames)
data_cleaned = filtered_data_df2.copy()


# In[30]:


# nEW copy of the filtered data
data_cleaned


# In[31]:


# Instantiate the MinMaxScaler with the desired feature range

data_scaler = data_cleaned.copy()
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit and transform the numerical features
data_scaler[numerical_features_list] = scaler.fit_transform(data_cleaned[numerical_features_list])

# Display the scaled data
data_scaler.head()



# In[32]:


df_clean = data_scaler.copy()

df_clean


# ## Machine Learning Implementation and Evaluation

# ### Algorithm selection 

# In[33]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df_clean.drop('Gender', axis=1)
y = df_clean['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Define candidate models
models = [
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC())
]

fold = 5
results = []
names = []

# Perform cross-validation for each model
for name, model in models:
    scores = cross_val_score(model, X, y, cv=fold, scoring='accuracy')
    results.append(scores)
    names.append(name)
    print(f"{name}: Accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")
    


# In[35]:


# Convert results to DataFrame for easy plotting
results_df = pd.DataFrame(results).T
results_df.columns = names

# Plotting the box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=results_df, notch=True, palette="Set2")

# Add data points
for i in range(results_df.shape[1]):
    y = results_df.iloc[:, i]
    x = np.random.normal(i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.6)

# Highlight the best model
medians = results_df.median().values
best_model_idx = np.argmax(medians)
best_model_name = results_df.columns[best_model_idx]

# Customize the plot
plt.title('Model Comparison using Cross-Validation')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0.5, 1.0)
plt.grid(True)

# Highlight the best model
plt.gca().get_xticklabels()[best_model_idx].set_color('red')
plt.gca().get_xticklabels()[best_model_idx].set_fontweight('bold')

plt.show()


# ### Best Model development 

# In[36]:


# Scale the features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  #### KNeighborsClassifier as the algorithm 

# In[37]:


# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Instantiate the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  


# In[38]:


# Train the model
knn_model.fit(X_train, y_train)


# In[39]:


# Make predictions
y_pred_knn = knn_model.predict(X_test)


# In[40]:


# Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn}')

# KNN Accuracy: 0.9980891719745223


# In[41]:


# Confusion matrix
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print('KNN Confusion Matrix:')
print(conf_matrix_knn)

# Plotting the confusion matrix for KNN
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()


# In[42]:


# Classification report
class_report_knn = classification_report(y_test, y_pred_knn)
print('KNN Classification Report:')
print(class_report_knn)



# In[43]:


df_clean


# In[44]:


# Convert the NumPy array to a DataFrame
test_results = pd.DataFrame(X_test, columns=X.columns)

# Reset the index
test_results.reset_index(drop=True, inplace=True)

# Add the predicted and actual gender columns
test_results['Predicted Gender'] = y_pred_knn
test_results['Actual Gender'] = y_test.values

# Plot scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=test_results, x='Age', y='PerfEval', hue='Predicted Gender', palette=['skyblue', 'pink'], style='Actual Gender')
plt.xlabel('Age')
plt.ylabel('Performance Evaluation')
plt.title('KNN Predictions')
plt.legend(title='Gender', loc='upper left')
plt.show()


# In[45]:


print("Shape of y_pred_knn:", y_pred_knn.shape)
print("Shape of test_results DataFrame:", y_test.shape)

# Shape of y_pred_knn: (1570,)
# Shape of test_results DataFrame: (1570, 9)


# In[ ]:





# In[46]:


###############################################
###############################################
###############################################
###############################################


# In[47]:


import eurostat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


# Set consistent color scheme
# MALE_COLOR = '#1f77b4'  # Blue
# FEMALE_COLOR = '#FF69B4'  # Hot Pink
# COLOR_PALETTE = sns.color_palette("husl", 8)

colors = sns.color_palette("husl", 4)
MALE_COLOR = colors[0]  # Primer color de la paleta
FEMALE_COLOR = colors[1]  # Segundo color de la paleta

EMP_LFS = colors[2] 
ACT =colors[3] 


# In[109]:


# Dataset code for Employment (LFS)
dataset_code = 'LFSI_EMP_A'

# Download the data
print(f"Downloading dataset: {dataset_code}")
df = eurostat.get_data_df(dataset_code)
df


# In[115]:


# Clean the data
df_cleaned = df.drop(columns=['2003', '2004', '2005', '2006', '2007', '2008', '2009'])

df_cleaned = df_cleaned.drop(columns=['freq'])
df_cleaned = df_cleaned.dropna()
df_cleaned


# In[111]:


# Rename the 'geo\\TIME_PERIOD' column to 'country'
df_cleaned = df_cleaned.rename(columns={'geo\\TIME_PERIOD': 'country'})
df_cleaned


# In[112]:


# EU countries
eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

# Create new dataset with only EU countries
df_eu = df_cleaned[df_cleaned['country'].isin(eu_countries)].copy()
df_eu


# In[113]:


# Remove rows where 'sex' is 'T' (Total)
df_filtered = df_eu[df_eu['sex'] != 'T']
df_filtered 


# In[54]:


# Identify numeric columns (years)
numeric_columns = df_filtered.select_dtypes(include=['float64', 'int64']).columns

# Function to calculate gender gap
def calculate_gender_gap(df, indicator, age_group='Y15-64', unit='PC_POP'):
    data = df[(df['indic_em'] == indicator) & (df['age'] == age_group) & (df['unit'] == unit)]
    
    print(f"\nShape of filtered data for {indicator}: {data.shape}")
    print(data.head())
    
    male_data = data[data['sex'] == 'M'][numeric_columns].mean()
    female_data = data[data['sex'] == 'F'][numeric_columns].mean()
    
    print("\nMale data:")
    print(male_data)
    print("\nFemale data:")
    print(female_data)
    
    return male_data - female_data


# In[55]:


# Calculate gender gaps
print("\nCalculating Employment Gap:")
employment_gap = calculate_gender_gap(df_filtered, 'EMP_LFS')
print("\nEmployment Gap:")
print(employment_gap)

print("\nCalculating Activity Gap:")
activity_gap = calculate_gender_gap(df_filtered, 'ACT')  # Changed from 'ACT_LFS' to 'ACT'
print("\nActivity Gap:")
print(activity_gap)


# In[56]:


# Create a comparison plot
plt.figure(figsize=(8, 5))
plt.plot(employment_gap.index, employment_gap.values, label='Employment Gap', marker='o', color=EMP_LFS)
plt.plot(activity_gap.index, activity_gap.values, label='Activity Gap', marker='s', color=ACT)
plt.title('Comparison of Employment and Activity Gender Gaps (2010-2023)')
plt.xlabel('Year')
plt.ylabel('Gender Gap (Percentage Points)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[57]:


# Additional analysis: Country-specific trends
def plot_country_trends(df, indicator, top_n=5):
    data = df[(df['indic_em'] == indicator) & (df['age'] == 'Y15-64') & (df['unit'] == 'PC_POP') & (df['sex'] == 'T')]
    latest_year = numeric_columns[-1]
    top_countries = data.nlargest(top_n, latest_year)['country'].tolist()
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Lista de colores
    
    plt.figure(figsize=(8, 5))
    for i, country in enumerate(top_countries):
        country_data = data[data['country'] == country]
        plt.plot(numeric_columns, country_data[numeric_columns].values[0], 
                 label=country, marker='o', color=colors[i])  # Definir color
    
    plt.title(f'Top {top_n} Countries: {indicator} Trend (2010-2023)')
    plt.xlabel('Year')
    plt.ylabel(f'{indicator} Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot country trends for employment and activity rates
plot_country_trends(df_eu, 'EMP_LFS')
plot_country_trends(df_eu, 'ACT')


# In[61]:


import nbformat
from nbconvert import PythonExporter

# Nombre de tu notebook actual
notebook_filename = 'CA 3 – Capstone Project Proposal.ipynb'

# Leer el notebook
with open(notebook_filename, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Crear un exportador de Python
exporter = PythonExporter()

# Exportar el notebook como script de Python
python_script, _ = exporter.from_notebook_node(notebook)

# Nombre del archivo de salida
output_filename = notebook_filename.replace('.ipynb', '.py')

# Guardar el script de Python
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(python_script)

print(f"El notebook ha sido convertido a {output_filename}")


# In[ ]:


################
################


# In[108]:


import eurostat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurar el estilo de las gráficas

MALE_COLOR = '#1f77b4'  # Azul
FEMALE_COLOR = '#FF69B4'  # Rosa

# Descargar los datasets
print("Descargando dataset de empleo...")
employment_data = eurostat.get_data_df('LFSI_EMP_A')
print("Descargando dataset de educación...")
education_data = eurostat.get_data_df('EDUC_UOE_ENRA03')

# Países de la UE
eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

# Función para preparar datos de empleo
def prepare_employment_data(df):
    df_cleaned = df.rename(columns={'geo\\TIME_PERIOD': 'country'})
    df_eu = df_cleaned[df_cleaned['country'].isin(eu_countries)].copy()
    df_filtered = df_eu[(df_eu['age'] == 'Y15-64') & (df_eu['unit'] == 'PC_POP') & 
                        (df_eu['sex'] != 'T') & (df_eu['indic_em'] == 'EMP_LFS')]
    
    year_columns = [col for col in df_filtered.columns if col.isdigit()]
    df_melted = df_filtered.melt(id_vars=['country', 'sex'], value_vars=year_columns, 
                                 var_name='year', value_name='employment_rate')
    return df_melted

# Función para preparar datos de educación
def prepare_education_data(df):
    df_cleaned = df.rename(columns={'geo\\TIME_PERIOD': 'country'})
    df_eu = df_cleaned[df_cleaned['country'].isin(eu_countries)].copy()
    df_tertiary = df_eu[df_eu['isced11'].isin(['ED5', 'ED6', 'ED7', 'ED8'])]
    df_filtered = df_tertiary[(df_tertiary['sex'] != 'T') & (df_tertiary['unit'] == 'NR')]
    
    year_columns = [col for col in df_filtered.columns if col.isdigit()]
    df_melted = df_filtered.melt(id_vars=['country', 'sex'], value_vars=year_columns, 
                                 var_name='year', value_name='education_enrollment')
    df_sum = df_melted.groupby(['country', 'sex', 'year'])['education_enrollment'].sum().reset_index()
    return df_sum

# Preparar los datos
employment_clean = prepare_employment_data(employment_data)
education_clean = prepare_education_data(education_data)

# Combinar los datasets
combined_data = pd.merge(employment_clean, education_clean, on=['country', 'sex', 'year'])

# Mostrar las primeras filas del dataset combinado
print("\nPrimeras filas del dataset combinado:")
print(combined_data.head(20))

# Calcular correlaciones entre tasas de empleo y matrículas en educación terciaria
years = combined_data['year'].unique()
correlations = {}

for year in years:
    year_data = combined_data[combined_data['year'] == year]
    corr = year_data['employment_rate'].corr(year_data['education_enrollment'])
    correlations[year] = corr

print("\nCorrelaciones entre tasas de empleo y matrículas en educación terciaria:")
print(correlations)

# Visualización de la relación entre empleo y educación para el último año disponible
latest_year = max(years)
latest_data = combined_data[combined_data['year'] == latest_year]

plt.figure(figsize=(12, 6))
sns.scatterplot(data=latest_data, x='education_enrollment', y='employment_rate', hue='sex', palette=[MALE_COLOR, FEMALE_COLOR])
plt.title(f'Relación entre Tasa de Empleo y Matrículas en Educación Terciaria ({latest_year})')
plt.xlabel('Matrículas en Educación Terciaria')
plt.ylabel('Tasa de Empleo (%)')
plt.legend(title='Sexo')
plt.show()

# Calcular la brecha de género en empleo y educación
gender_gap = combined_data.pivot_table(
    values=['employment_rate', 'education_enrollment'], 
    index=['country', 'year'], 
    columns='sex', 
    aggfunc='first'
).reset_index()

gender_gap['emp_gap'] = gender_gap[('employment_rate', 'M')] - gender_gap[('employment_rate', 'F')]
gender_gap['edu_gap'] = gender_gap[('education_enrollment', 'M')] - gender_gap[('education_enrollment', 'F')]

# Visualizar la brecha de género
plt.figure(figsize=(12, 6))
sns.heatmap(gender_gap.pivot(index='country', columns='year', values='emp_gap'), cmap='coolwarm', center=0)
plt.title('Brecha de Género en Tasas de Empleo por País y Año')
plt.xlabel('Año')
plt.ylabel('País')
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(gender_gap.pivot(index='country', columns='year', values='edu_gap'), cmap='coolwarm', center=0)
plt.title('Brecha de Género en Matrículas de Educación Terciaria por País y Año')
plt.xlabel('Año')
plt.ylabel('País')
plt.show()

print("Análisis completado. Por favor, revisa las gráficas para la representación visual de los datos.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




