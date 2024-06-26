---
title: Pandas DataFrames
teaching: 15
exercises: 15
objectives:
- Select individual values from a Pandas dataframe.
- Select entire rows or entire columns from a dataframe.
- Select a subset of both rows and columns from a dataframe in a single operation.
- Select a subset of a dataframe by a single Boolean criterion.
questions:
- How can I do statistical analysis of tabular data?
keypoints:
- Use `DataFrame.iloc[..., ...]` to select values by integer location.
- Use `:` on its own to mean all columns or all rows.
- Select multiple columns or rows using `DataFrame.loc` and a named slice.
- Result of slicing can be used in further operations.
- Use comparisons to select data based on value.
- Select values or NaN using a Boolean mask.
---

## Pandas DataFrames/Series

A [DataFrame][pandas-dataframe] is a collection of [Series][pandas-series];
The DataFrame is the way Pandas represents a table, and Series is the data-structure
Pandas use to represent a column.

Pandas is built on top of the [Numpy][numpy] library, which in practice means that
most of the methods defined for Numpy Arrays apply to Pandas Series/DataFrames.

What makes Pandas so attractive is the powerful interface to access individual records
of the table, proper handling of missing values, and relational-databases operations
between DataFrames.

The basic steps to load and preview a dataset using the Pandas library in Python. The dataset being used is the "Gapminder" dataset, which contains information about various socioeconomic indicators for different countries over time.

```python
import pandas as pd

path = "https://raw.githubusercontent.com/mesfind/datasets/master/gapminder_tidy.csv"
df= pd.read_csv(path)
df.head()
```

```output
       Country  Year  fertility    life  population  child_mortality     gdp      region
0  Afghanistan  1964      7.671  33.639  10474903.0            339.7  1182.0  South Asia
1  Afghanistan  1965      7.671  34.152  10697983.0            334.1  1182.0  South Asia
2  Afghanistan  1966      7.671  34.662  10927724.0            328.7  1168.0  South Asia
3  Afghanistan  1967      7.671  35.170  11163656.0            323.3  1173.0  South Asia
4  Afghanistan  1968      7.671  35.674  11411022.0            318.1  1187.0  South Asia

```

With an additional parameter  in the `pd.read_csv()`` function, it  is possible to load the same  dataset. Using the `index_col` parameter allows you to specify which column should be used as the index for the DataFrame. In this case, setting index_col='Country' tells Pandas to use the "Country" column as the index:

```python
import pandas as pd

path = "https://raw.githubusercontent.com/mesfind/datasets/master/gapminder_tidy.csv"
df= pd.read_csv(path, index_col='Country')
df.head()
```

```output
             Year  fertility    life  population  child_mortality     gdp      region
Country                                                                              
Afghanistan  1964      7.671  33.639  10474903.0            339.7  1182.0  South Asia
Afghanistan  1965      7.671  34.152  10697983.0            334.1  1182.0  South Asia
Afghanistan  1966      7.671  34.662  10927724.0            328.7  1168.0  South Asia
Afghanistan  1967      7.671  35.170  11163656.0            323.3  1173.0  South Asia
Afghanistan  1968      7.671  35.674  11411022.0            318.1  1187.0  South Asia
```

Printing the column names is a common first step in data exploration, as it gives you a high-level overview of the data structure and helps you navigate the dataset more effectively.

```python
print(df.columns)
```

```output
Index(['Country', 'Year', 'fertility', 'life', 'population', 'child_mortality',
       'gdp', 'region'],
      dtype='object')

```
This is a useful way to quickly see the available data columns in the dataset, which can help you understand the types of information contained in the data and plan your analysis accordingly.

## Selecting values

To access a value at the position `[i,j]` of a DataFrame, we have two options, depending on
what is the meaning of `i` in use.
Remember that a DataFrame provides an *index* as a way to identify the rows of the table;
a row, then, has a *position* inside the table as well as a *label*, which
uniquely identifies its *entry* in the DataFrame.


- Can specify location by numerical index analogously to 2D version of character selection in strings.

## Use `DataFrame.iloc[..., ...]` to select values by their (entry) position

```python
print(df.iloc[0,1])
```

```output
7.671
```


```python
print(df.iloc[[0,1]])
```

```
             Year  fertility    life  population  child_mortality     gdp      region
Country                                                                              
Afghanistan  1964      7.671  33.639  10474903.0            339.7  1182.0  South Asia
Afghanistan  1965      7.671  34.152  10697983.0            334.1  1182.0  South Asia
```

## Use `DataFrame.loc[..., ...]` to select values by their (entry) label.

- Can specify location by row and/or column name.

```python
print(df.loc["Ethiopia", "gdp"])
```

```output
Country
Ethiopia     698.0
Ethiopia     727.0
Ethiopia     746.0
Ethiopia     777.0
Ethiopia     800.0
Ethiopia     810.0
Ethiopia     778.0
Ethiopia     792.0
Ethiopia     803.0
...
Ethiopia     863.0
Ethiopia     931.0
Ethiopia     986.0
Ethiopia    1081.0
Ethiopia    1171.0
Ethiopia    1240.0
Ethiopia    1336.0
Name: gdp, dtype: float64

```

## Use `:` on its own to mean all columns or all rows.

- Just like Python's usual slicing notation.

```python
print(df.loc["Ethiopia", :])
```

```output
         Year  fertility    life  population  child_mortality     gdp              region
Country                                                                                   
Ethiopia  1964      6.867  40.200  24846785.0           255.99   698.0  Sub-Saharan Africa
Ethiopia  1965      6.864  39.000  25480202.0           248.90   727.0  Sub-Saharan Africa
Ethiopia  1966      6.867  40.000  26128435.0           242.00   746.0  Sub-Saharan Africa
Ethiopia  1967      6.880  42.027  26790992.0           241.50   777.0  Sub-Saharan Africa
Ethiopia  1968      6.903  42.357  27476546.0           241.00   800.0  Sub-Saharan Africa
Ethiopia  1969      6.937  42.663  28197484.0           241.20   810.0  Sub-Saharan Africa
Ethiopia  1970      6.978  42.949  28959382.0           241.10   778.0  Sub-Saharan Africa
Ethiopia  1971      7.020  43.219  29777985.0           241.20   792.0  Sub-Saharan Africa
Name: Albania, dtype: float64
```

- Would get the same result printing `data.loc["Albania"]` (without a second index).

```python
print(df.loc[:, "gdp"])
```

```output
Country
Afghanistan    1182.0
Afghanistan    1182.0
Afghanistan    1168.0
Afghanistan    1173.0
Afghanistan    1187.0
...               ...  
Åland             NaN
Åland             NaN
Åland             NaN
Åland             NaN
Åland             NaN
Name: gdp, Length: 10111, dtype: float64
```

- Would get the same result printing `df["gdp"]`
- Also get the same result printing `df.gdp` (not recommended, because easily confused with `.` notation for methods)

## Select multiple columns or rows using `DataFrame.loc` and a named slice.

```python
print(df.loc['Ethiopia':'Somalia', 'gdp':'region'])
```

```output
            gdp              region
Country                            
Ethiopia  698.0  Sub-Saharan Africa
Ethiopia  727.0  Sub-Saharan Africa
Ethiopia  746.0  Sub-Saharan Africa
Ethiopia  777.0  Sub-Saharan Africa
Ethiopia  800.0  Sub-Saharan Africa
...         ...                 ...
Somalia   615.0  Sub-Saharan Africa
Somalia   614.0  Sub-Saharan Africa
Somalia   614.0  Sub-Saharan Africa
Somalia   616.0  Sub-Saharan Africa
Somalia   619.0  Sub-Saharan Africa

[5450 rows x 2 columns]

```

In the above code, we discover that **slicing using `loc` is inclusive at both
ends**, which differs from **slicing using `iloc`**, where slicing indicates
everything up to but not including the final index.

## Result of slicing can be used in further operations.

- Usually don't just print a slice.
- All the statistical operators that work on entire dataframes
  work the same way on slices.
- E.g., calculate max of a slice.

```python
print(df.loc['Ethiopia':'Somalia', 'gdp'].max())
```

```output
165564.0
```

```python
print(df.loc['Ethiopia':'Somalia', 'gdp'].min())
```

```output
142.0
```

## Use comparisons to select data based on value.

- Comparison is applied element by element.
- Returns a similarly-shaped dataframe of `True` and `False`.

```python
# Use a subset of data to keep output readable.
subset = df.loc['Ethiopia':'Somalia', 'gdp']
print('Subset of data:\n', subset)

# Which values were greater than 10000 ?
print('\nWhere are values large?\n', subset > 10000)
```

```output
Subset of data:
 Country
Ethiopia    698.0
Ethiopia    727.0
Ethiopia    746.0
Ethiopia    777.0
Ethiopia    800.0
            ...  
Somalia     615.0
Somalia     614.0
Somalia     614.0
Somalia     616.0
Somalia     619.0
Name: gdp, Length: 5450, dtype: float64

Where are values large?
 Country
Ethiopia    False
Ethiopia    False
Ethiopia    False
Ethiopia    False
Ethiopia    False
            ...  
Somalia     False
Somalia     False
Somalia     False
Somalia     False
Somalia     False
Name: gdp, Length: 5450, dtype: bool
```

## Select values or NaN using a Boolean mask.

- A frame full of Booleans is sometimes called a *mask* because of how it can be used.

```python
mask = subset > 10000
print(subset[mask])
```

```output
Country
Finland     12389.0
Finland     13006.0
Finland     13269.0
Finland     13477.0
Finland     13726.0
             ...   
Slovenia    28157.0
Slovenia    28377.0
Slovenia    28492.0
Slovenia    27682.0
Slovenia    27368.0
Name: gdp, Length: 1964, dtype: float64
```

- Get the value where the mask is true, and NaN (Not a Number) where it is false.
- Useful because NaNs are ignored by operations like max, min, average, etc.

```python
print(subset[subset > 10000].describe())
```

```output
count      1964.000000
mean      27467.373727
std       21749.445029
min       10003.000000
25%       14195.750000
50%       20403.500000
75%       32395.250000
max      165564.000000
Name: gdp, dtype: float64
```

## Selecting Numerical Columns

By creating a separate DataFrame with only the numerical columns, you can:

1. Analyze the numerical data more effectively, without having to worry about non-numerical columns.
2. Perform statistical operations, such as calculating means, standard deviations, and correlations, on the numerical data.
3. Use the numerical data as input for machine learning models or other data analysis techniques that require numerical inputs.

The following  code snippet identifies the numerical columns in the DataFrame and creates a new DataFrame containing only those columns. The `select_dtypes(include=['number'])` method is used to filter the DataFrame and keep only the columns with numerical data types, such as integers and floats. This is a common data preprocessing step, as it allows you to focus on the quantitative aspects of the data and perform numerical analysis more easily.

```python
# numerical columns
numerical_df = df.select_dtypes(include=['number'])
numerical_df.head()
```

```output
   Year  fertility    life  population  child_mortality     gdp
0  1964      7.671  33.639  10474903.0            339.7  1182.0
1  1965      7.671  34.152  10697983.0            334.1  1182.0
2  1966      7.671  34.662  10927724.0            328.7  1168.0
3  1967      7.671  35.170  11163656.0            323.3  1173.0
4  1968      7.671  35.674  11411022.0            318.1  1187.0
```


Checking for missing data is a crucial step in data exploration, as it helps you identify potential issues or limitations in the dataset. Addressing missing data can involve techniques like imputation, dropping rows or columns with missing values, or using specialized methods for handling missing data in your analysis.

```python
numerical_df.isnull().sum()
```

```output
Year                  0
fertility            11
life                  0
population            3
child_mortality     901
gdp                1111
dtype: int64
```
The output shows the count of missing values (NaN) for each column in the DataFrame. Let's discuss this in more detail:

- `Year`: 0 missing values, indicating that the year data is complete.
- `fertility`: 11 missing values, suggesting that the fertility data has some gaps.
- `life`: 0 missing values, meaning the life expectancy data is complete.
- `population`: 3 missing values, so the population data has a few gaps.
- `child_mortality`: 901 missing values, which is a significant amount of missing data for the child mortality column.
- `gdp`: 1111 missing values, indicating that the GDP data has a large number of missing entries.


## Handling Missing Valus

Some common approaches to handling missing data include:

- Imputation: Filling in the missing values using techniques like mean, median, or more advanced methods.
- Dropping rows or columns with missing data, if the amount of missing data is not too high.
- Using specialized methods for handling missing data, such as those available in machine learning libraries.

Knowing the extent of missing data in each column will help you make informed decisions about how to best prepare the data for your analysis.

### Imputing missing with Median

Using the median as the imputation method is a simple and robust approach, as it is less sensitive to outliers compared to using the mean. 
```python
numerical_df = numerical_df.fillna(numerical_df.median())
numerical_df.isnull().sum()
```

```output
Year               0
fertility          0
life               0
population         0
child_mortality    0
gdp                0
dtype: int64
```

However, it's important to note that imputing missing values can introduce some bias in the data, especially if the missing values are not missing at random. In such cases, more advanced imputation techniques or domain-specific knowledge may be required to handle the missing data more effectively.

### Dropping  missing data,

Dropping rows with missing values is a common approach to handling missing data, especially when the amount of missing data is not too high. This method ensures that your analysis is performed on a complete dataset, without the need for imputation. However, it's important to be cautious when dropping a significant amount of data, as it can lead to a loss of information and potentially biased results.

```
import pandas as pd

path = "https://raw.githubusercontent.com/mesfind/datasets/master/gapminder_tidy.csv"
df = pd.read_csv(path, index_col='Country')
numerical_df2 = df.select_dtypes(include=['number'])
numerical_df2 = numerical_df.dropna(inplace=False)
print(numerical_df2.isnull().sum())
```

```output
Year               0
fertility          0
life               0
population         0
child_mortality    0
gdp                0
dtype: int64

```

## Outlier Detections

### Interquartile Range (IQR) method

To treat outliers identified using the Interquartile Range (IQR) method, you have several options depending on the context and the nature of your data. Common methods include removing the outliers, replacing them with a statistical value (such as the median), or capping them at the bounds of the IQR range. Below are examples of each approach:

```python
# Calculate the IQR for each numerical column
Q1 = numerical_df.quantile(0.25)
Q3 = numerical_df.quantile(0.75)
IQR = Q3 - Q1
# Determine outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (numerical_df < lower_bound) | ( numerical_df > upper_bound)
print("Potential outliers based on IQR:\n",outliers)
```

```output
Potential outliers based on IQR:
               Year  fertility   life  population  child_mortality    gdp
Country                                                                 
Afghanistan  False      False  False       False             True  False
Afghanistan  False      False  False       False             True  False
Afghanistan  False      False  False       False             True  False
Afghanistan  False      False  False       False             True  False
Afghanistan  False      False  False       False             True  False
...            ...        ...    ...         ...              ...    ...
Zimbabwe     False      False  False       False            False  False
Zimbabwe     False      False  False       False            False  False
Zimbabwe     False      False  False       False            False  False
Zimbabwe     False      False  False       False            False  False
Zimbabwe     False      False  False       False            False  False

[8836 rows x 6 columns]
```

- Calculates the sum of the outliers DataFrame (or Series) using the `sum()` method, and stores the result in the outlier_counts variable.
- The `outlier_counts` variable displays the count of potential outliers in each column of the original DataFrame.

```python
outlier_counts = outliers.sum()
print("Count of potential outliers in each column:\n", outlier_counts)
```


```python
Count of potential outliers in each column:
 Year                 0
fertility             0
life                 10
population         1208
child_mortality     202
gdp                 723
dtype: int64
```

This output indicates that there are 10 potential outliers in the "life", 1208 in "population",  202 in "child_mortality"  and 723 in "gdp" columns in `numerical_df`  DataFrame.


Identifying  the rows in the `numerical_df` DataFrame that contain potential outliers can be useful for further data exploration, cleaning, and analysis, as you can now focus on these potentially problematic rows and investigate the outliers in more detail.

```python
outlier_rows = numerical_df[outliers.any(axis=1)]
print("Rows with potential outliers:\n",outlier_rows)
```

```
Rows with potential outliers:
              Year  fertility    life  population  child_mortality     gdp
Country                                                                  
Afghanistan  1964      7.671  33.639  10474903.0            339.7  1182.0
Afghanistan  1965      7.671  34.152  10697983.0            334.1  1182.0
Afghanistan  1966      7.671  34.662  10927724.0            328.7  1168.0
Afghanistan  1967      7.671  35.170  11163656.0            323.3  1173.0
Afghanistan  1968      7.671  35.674  11411022.0            318.1  1187.0
...           ...        ...     ...         ...              ...     ...
Vietnam      2009      1.843  75.344  86901173.0             25.5  4260.0
Vietnam      2010      1.820  75.490  87848445.0             24.8  4486.0
Vietnam      2011      1.794  75.641  88791996.0             24.2  4717.0
Vietnam      2012      1.768  75.793  89730274.0             23.5  4912.0
Vietnam      2013      1.743  75.945  90656550.0             22.9  5125.0

[2060 rows x 6 columns]
```



### Capping Outliers at the Bounds of the IQR Range

Capping outliers involves adjusting extreme values in a dataset to the nearest boundary of the interquartile range (IQR). This technique effectively reduces the impact of outliers, ensuring that statistical analyses are not unduly influenced by anomalous data points. By capping values at the IQR bounds, researchers maintain data integrity while enhancing the robustness of their statistical models.

```python
# Calculate the IQR for each numerical column
Q1 = numerical_df.quantile(0.25)
Q3 = numerical_df.quantile(0.75)
IQR = Q3 - Q1
# Determine outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Cap outliers at the bounds of the IQR range
capped_df = numerical_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

print("DataFrame after capping outliers:\n", capped_df)
```

```python
DataFrame after capping outliers:
              Year  fertility    life  population  child_mortality     gdp
Country                                                                  
Afghanistan  1964      7.671  33.639  10474903.0            285.3  1182.0
Afghanistan  1965      7.671  34.152  10697983.0            285.3  1182.0
Afghanistan  1966      7.671  34.662  10927724.0            285.3  1168.0
Afghanistan  1967      7.671  35.170  11163656.0            285.3  1173.0
Afghanistan  1968      7.671  35.674  11411022.0            285.3  1187.0
...           ...        ...     ...         ...              ...     ...
Zimbabwe     2009      3.792  51.234  12473992.0             97.3  1352.0
Zimbabwe     2010      3.721  53.684  12571454.0             95.1  1484.0
Zimbabwe     2011      3.643  56.040  12754378.0             92.0  1626.0
Zimbabwe     2012      3.564  58.142  13013678.0             86.7  1750.0
Zimbabwe     2013      3.486  59.871  13327925.0             83.3  1773.0

[8836 rows x 6 columns]
```


## Logarithmic Transformation

Logarithmic transformation is useful when the data spans several orders of magnitude. It can compress the range and make the distribution more symmetric.

```python
import numpy as np
cols = ['fertility', 'life', 'population',
        'child_mortality', 'gdp']
log_transformed_df = np.log1p(numerical_df[cols]) # log1p is used to handle zero values
log_transformed_df['year'] = numerical_df["Year"]
print("DataFrame after logarithmic transformation:\n", log_transformed_df)

```

```output
DataFrame after logarithmic transformation:
              fertility      life  population  child_mortality       gdp  year
Country                                                                      
Afghanistan   2.159984  3.544980   16.164493         5.831002  7.075809  1964
Afghanistan   2.159984  3.559682   16.185566         5.814429  7.075809  1965
Afghanistan   2.159984  3.574086   16.206814         5.798183  7.063904  1966
Afghanistan   2.159984  3.588230   16.228174         5.781669  7.068172  1967
Afghanistan   2.159984  3.602068   16.250090         5.765505  7.080026  1968
...                ...       ...         ...              ...       ...   ...
Zimbabwe      1.566948  3.955734   16.339156         4.588024  7.210080  2009
Zimbabwe      1.552021  4.001571   16.346939         4.565389  7.303170  2010
Zimbabwe      1.535361  4.043753   16.361385         4.532599  7.394493  2011
Zimbabwe      1.518199  4.079941   16.381512         4.473922  7.467942  2012
Zimbabwe      1.500961  4.108757   16.405372         4.434382  7.480992  2013

[8836 rows x 6 columns]
```

> ## Square Root Transformation
> Square root transformation is another method to reduce the impact of large values. It's less aggressive than logarithmic transformation. apply square root transformation on `numerical_df`
>
> > ## Solution
> > ```python
> > import numpy as np
> > cols = ['fertility', 'life', 'population',
> >       'child_mortality', 'gdp']
> > sqrt_transformed_df = np.sqrt(numerical_df[cols])
> > sqrt_transformed_df['year'] = numerical_df.Year
> > print("DataFrame after square transformation:\n", sqrt_transformed_df)
> > ```
> > ```output
> > DataFrame after square transformation:
> >               fertility      life   population  child_mortality        gdp  year
> > Country                                                                        
> > Afghanistan   2.769657  5.799914  3236.495481        18.430952  34.380227  1964
> > Afghanistan   2.769657  5.843971  3270.777125        18.278403  34.380227  1965
> > Afghanistan   2.769657  5.887444  3305.710816        18.130085  34.176015  1966
> > Afghanistan   2.769657  5.930430  3341.205770        17.980545  34.249088  1967
> > Afghanistan   2.769657  5.972772  3378.020426        17.835358  34.452866  1968
> > ...                ...       ...          ...              ...        ...   ...
> > Zimbabwe      1.947306  7.157793  3531.853904         9.864076  36.769553  2009
> > Zimbabwe      1.928989  7.326937  3545.624628         9.751923  38.522721  2010
> > Zimbabwe      1.908664  7.485987  3571.327204         9.591663  40.323690  2011
> > Zimbabwe      1.887856  7.625090  3607.447574         9.311283  41.833001  2012
> > Zimbabwe      1.867083  7.737635  3650.743075         9.126883  42.107007  2013
> > 
> > [8836 rows x 6 columns]
> > ```
> >
> {: .solution}
{.challenge}

> ## Box-Cox Transformation
> Box-Cox transformation is a more flexible method that can transform data to follow a normal distribution. It requires the data to be positive.  Apply square root transformation on `numerical_df`
> 
> > ```python
> > from scipy import stats
> > # Apply Box-Cox transformation to each column
> > boxcox_transformed_df = numerical_df[cols].apply(lambda x: stats.boxcox(x + 1)[0] if (x > 0).all() else x)  # Adding 1 to handle zero values
> > boxcox_transformed_df['year'] = numerical_df.Year
> > print("DataFrame after Box-Cox transformation:\n", boxcox_transformed_df)
> > ```
> >
> > ```out
> > DataFrame after Box-Cox transformation:
> >               fertility          life  population  child_mortality       gdp  year
> > Country                                                                          
> > Afghanistan   2.765851   3047.980770   20.015617         8.549761  6.387016  1964
> > Afghanistan   2.765851   3163.228929   20.047480         8.515674  6.387016  1965
> > Afghanistan   2.765851   3280.372721   20.079625         8.482328  6.377349  1966
> > Afghanistan   2.765851   3399.623456   20.111957         8.448500  6.380815  1967
> > Afghanistan   2.765851   3520.485139   20.145150         8.415455  6.390439  1968
> > ...                ...           ...         ...              ...       ...   ...
> > Zimbabwe      1.871283   8596.829323   20.280233         6.177878  6.495807  2009
> > Zimbabwe      1.850239   9651.393648   20.292051         6.137981  6.570980  2010
> > Zimbabwe      1.826833  10735.778689   20.313994         6.080382  6.644526  2011
> > Zimbabwe      1.802813  11762.692982   20.344579         5.977890  6.703535  2012
> > Zimbabwe      1.778777  12650.185919   20.380859         5.909245  6.714005  2013
> > 
> > [8836 rows x 6 columns]
> > 
> > ```
> {: .solution}
{: .challenge}


## Descriptive Analysis

```python
capped_df.describe()
```

```
     Year    fertility         life    population  child_mortality           gdp
count  8836.000000  8836.000000  8836.000000  8.836000e+03      8836.000000   8836.000000
mean   1988.623133     4.096088    63.586344  1.161399e+07        82.351941  10808.059091
std      14.405538     2.042115    11.278558  1.335891e+07        77.425250  11011.999498
min    1964.000000     0.836000    29.273125  3.680100e+04         2.000000    142.000000
25%    1976.000000     2.164750    55.063750  1.600499e+06        19.800000   2222.250000
50%    1989.000000     3.794000    66.661000  5.408103e+06        54.500000   6299.000000
75%    2001.000000     6.039250    72.257500  1.642182e+07       126.000000  15639.500000
```

## Group By: split-apply-combine

> Learners often struggle here, many may not work with financial data and concepts so they
> find the example concepts difficult to get their head around. The biggest problem
> though is the line generating the wealth_score, this step needs to be talked through
> throughly:
> * It uses implicit conversion between boolean and float values which  has not been covered in the course so far. 
> * The axis=1 argument needs to be explained clearly.
>
{: .instructor}

Pandas vectorizing methods and grouping operations are features that provide users
much flexibility to analyse their data.

For instance, let's say we want to have a clearer view on how the European countries
split themselves according to their GDP.

1. We may have a glance by splitting the countries in two groups during the years surveyed,
  those who presented a GDP *higher* than the European average and those with a *lower* GDP.
2. We then estimate a *wealthy score* based on the historical (from 1962 to 2007) values,
  where we account how many times a country has participated in the groups of *lower* or *higher* GDP

```python
mask_higher = data > data.mean()
wealth_score = mask_higher.aggregate('sum', axis=1) / len(data.columns)
print(wealth_score)
```

```output
country
Albania                   0.000000
Austria                   1.000000
Belgium                   1.000000
Bosnia and Herzegovina    0.000000
Bulgaria                  0.000000
Croatia                   0.000000
Czech Republic            0.500000
Denmark                   1.000000
Finland                   1.000000
France                    1.000000
Germany                   1.000000
Greece                    0.333333
Hungary                   0.000000
Iceland                   1.000000
Ireland                   0.333333
Italy                     0.500000
Montenegro                0.000000
Netherlands               1.000000
Norway                    1.000000
Poland                    0.000000
Portugal                  0.000000
Romania                   0.000000
Serbia                    0.000000
Slovak Republic           0.000000
Slovenia                  0.333333
Spain                     0.333333
Sweden                    1.000000
Switzerland               1.000000
Turkey                    0.000000
United Kingdom            1.000000
dtype: float64
```

Finally, for each group in the `wealth_score` table, we sum their (financial) contribution
across the years surveyed using chained methods:

```python
print(data.groupby(wealth_score).sum())
```

```output
          gdpPercap_1952  gdpPercap_1957  gdpPercap_1962  gdpPercap_1967  \
0.000000    36916.854200    46110.918793    56850.065437    71324.848786   
0.333333    16790.046878    20942.456800    25744.935321    33567.667670   
0.500000    11807.544405    14505.000150    18380.449470    21421.846200   
1.000000   104317.277560   127332.008735   149989.154201   178000.350040   

          gdpPercap_1972  gdpPercap_1977  gdpPercap_1982  gdpPercap_1987  \
0.000000    88569.346898   104459.358438   113553.768507   119649.599409   
0.333333    45277.839976    53860.456750    59679.634020    64436.912960   
0.500000    25377.727380    29056.145370    31914.712050    35517.678220   
1.000000   215162.343140   241143.412730   263388.781960   296825.131210   

          gdpPercap_1992  gdpPercap_1997  gdpPercap_2002  gdpPercap_2007  
0.000000    92380.047256   103772.937598   118590.929863   149577.357928  
0.333333    67918.093220    80876.051580   102086.795210   122803.729520  
0.500000    36310.666080    40723.538700    45564.308390    51403.028210  
1.000000   315238.235970   346930.926170   385109.939210   427850.333420
```

:::::::::::::::::::::::::::::::::::::::  challenge

> ## Selection of Individual Values
> 
> Assume Pandas has been imported into your notebook
> and the Gapminder GDP data for Europe has been loaded:
> 
> ```python
> import pandas as pd
> 
> data_africa = pd.read_csv('data/gapminder_tidy.csv', index_col='country'=='Africa')
> ```
> 
> Write an expression to find the Per Capita GDP of Serbia in 2007.
> 
> > ## Solution
> > 
> > The selection can be done by using the labels for both the row ("Serbia") and the column ("gdpPercap\_2007"):
> > 
> > ```python
> > print(data_europe.loc['Sudan', 'gdpPercap_2007'])
> > ```
> > 
> > The output is
> > 
> > ```output
> > 9786.534714
> > ```
> {: .solution}
{: .challenge}


> ## Extent of Slicing
>
> 1. Do the two statements below produce the same output?
> 2. Based on this,
>  what rule governs what is included (or not) in numerical slices and named slices in Pandas?
>
> ```python
> print(data_africa.iloc[0:2, 0:2])
> print(data_africa.loc['Somalia':'Ethiopia', 'gdpPercap_1952':'gdpPercap_1962'])
> ```
>
> > ## Solution
> > 
> > No, they do not produce the same output! The output of the first statement is:
> > 
> > ```output
> >         gdpPercap_1952  gdpPercap_1957
> > country                                
> > Ethiopia     1601.056136     1942.284244
> > Somalia     6137.076492     8842.598030
> > ```
> > 
> > The second statement gives:
> > 
> > ```output
> >         gdpPercap_1952  gdpPercap_1957  gdpPercap_1962
> > country                                                
> > Albania     1601.056136     1942.284244     2312.888958
> > Austria     6137.076492     8842.598030    10750.721110
> > Belgium     8343.105127     9714.960623    10991.206760
> > ```
> > 
> > Clearly, the second statement produces an additional column and an additional row compared to the first statement.  
> > What conclusion can we draw? We see that a numerical slice, 0:2, *omits* the final index (i.e. index 2)
> > in the range provided,
> > while a named slice, 'gdpPercap\_1952':'gdpPercap\_1962', *includes* the final element.
> {: .solution}
{: .challenge}

:::::::::::::::::::::::::::::::::::::::  challenge

> ## Reconstructing Data
> 
> Explain what each line in the following short program does:
> what is in `first`, `second`, etc.?
> 
> ```python
> first = pd.read_csv('data/gapminder.csv', index_col='country')
> second = first[first['continent'] == 'Americas']
> third = second.drop('Puerto Rico')
> fourth = third.drop('continent', axis = 1)
> fourth.to_csv('result.csv')
> ```
> > ## Solution
> > 
> > Let's go through this piece of code line by line.
> > 
> > ```python
> > first = pd.read_csv('data/gapminder_all.csv', index_col='country')
> > ```
> > 
> > This line loads the dataset containing the GDP data from all countries into a dataframe called
> > `first`. The `index_col='country'` parameter selects which column to use as the
> > row labels in the dataframe.
> > 
> > ```python
> > second = first[first['continent'] == 'Americas']
> > ```
> > 
> > This line makes a selection: only those rows of `first` for which the 'continent' column matches
> > 'Americas' are extracted. Notice how the Boolean expression inside the brackets,
> > `first['continent'] == 'Americas'`, is used to select only those rows where the expression is true.
> > Try printing this expression! Can you print also its individual True/False elements?
> > (hint: first assign the expression to a variable)
> > 
> > ```python
> > third = second.drop('Puerto Rico')
> > ```
> > 
> > As the syntax suggests, this line drops the row from `second` where the label is 'Puerto Rico'. The
> > resulting dataframe `third` has one row less than the original dataframe `second`.
> > 
> > ```python
> > fourth = third.drop('continent', axis = 1)
> > ```
> > 
> > Again we apply the drop function, but in this case we are dropping not a row but a whole column.
> > To accomplish this, we need to specify also the `axis` parameter (we want to drop the second column
> > which has index 1).
> > 
> > ```python
> > fourth.to_csv('result.csv')
> > ```
> > 
> > The final step is to write the data that we have been working on to a csv file. Pandas makes this easy
> > with the `to_csv()` function. The only required argument to the function is the filename. Note that the
> > file will be written in the directory from which you started the Jupyter or Python session.
> {: .solution}
{: challenge}



> ## Selecting Indices
> 
> Explain in simple terms what `idxmin` and `idxmax` do in the short program below.
> When would you use these methods?
>
> ```python
> data = pd.read_csv('data/gapminder_gdp_europe.csv', index_col='country')
> print(data.idxmin())
> print(data.idxmax())
> ```
>
>
> > ## Solution
> > 
> > For each column in `data`, `idxmin` will return the index value corresponding to each column's minimum;
> > `idxmax` will do accordingly the same for each column's maximum value.
> > 
> > You can use these functions whenever you want to get the row index of the minimum/maximum value and not the actual minimum/maximum value.
> {: .solution}
{: .challenge}



> ## Practice with Selection
>
> Assume Pandas has been imported and the Gapminder GDP data for Europe has been loaded.
> Write an expression to select each of the following:
> 
> 1. GDP per capita for all countries in 1982.
> 2. GDP per capita for Denmark for all years.
> 3. GDP per capita for all countries for years *after* 1985.
> 4. GDP per capita for each country in 2007 as a multiple of
>  GDP per capita for that country in 1952.
>
> > ## Solution
> > 
> > 1:
> > 
> > ```python
> > data['gdpPercap_1982']
> > ```
> > 
> > 2:
> > 
> > ```python
> > data.loc['Denmark',:]
> > ```
> > 
> > 3:
> > 
> > ```python
> > data.loc[:,'gdpPercap_1985':]
> > ```
> > 
> > Pandas is smart enough to recognize the number at the end of the column label and does not give you an error, although no column named `gdpPercap_1985` actually exists. This is useful if new columns are added to the CSV file later.
> > 
> > 4:
> > 
> > ```python
> > data['gdpPercap_2007']/data['gdpPercap_1952']
> > ```
> {: .solution}
{: .challenge}


> ## Many Ways of Access
> 
> There are at least two ways of accessing a value or slice of a DataFrame: by name or index.
> However, there are many others. For example, a single column or row can be accessed either as a `DataFrame`
> or a `Series` object.
> 
> Suggest different ways of doing the following operations on a DataFrame:
>
> 1. Access a single column
> 2. Access a single row
> 3. Access an individual DataFrame element
> 4. Access several columns
> 5. Access several rows
> 6. Access a subset of specific rows and columns
> 7. Access a subset of row and column ranges
>
> >  ## Solution
> > 
> > 1\. Access a single column:
> > 
> > ```python
> > # by name
> > data["col_name"]   # as a Series
> > data[["col_name"]] # as a DataFrame
> > 
> > # by name using .loc
> > data.T.loc["col_name"]  # as a Series
> > data.T.loc[["col_name"]].T  # as a DataFrame
> > 
> > # Dot notation (Series)
> > data.col_name
> > 
> > # by index (iloc)
> > data.iloc[:, col_index]   # as a Series
> > data.iloc[:, [col_index]] # as a DataFrame
> > 
> > # using a mask
> > data.T[data.T.index == "col_name"].T
> > ```
> > 
> > 2\. Access a single row:
> > 
> > ```python
> > # by name using .loc
> > data.loc["row_name"] # as a Series
> > data.loc[["row_name"]] # as a DataFrame
> > 
> > # by name
> > data.T["row_name"] # as a Series
> > data.T[["row_name"]].T # as a DataFrame
> > 
> > # by index
> > data.iloc[row_index]   # as a Series
> > data.iloc[[row_index]]   # as a DataFrame
> > 
> > # using mask
> > data[data.index == "row_name"]
> > ```
> > 
> > 3\. Access an individual DataFrame element:
> > 
> > ```python
> > # by column/row names
> > data["column_name"]["row_name"]         # as a Series
> > 
> > data[["col_name"]].loc["row_name"]  # as a Series
> > data[["col_name"]].loc[["row_name"]]  # as a DataFrame
> > 
> > data.loc["row_name"]["col_name"]  # as a value
> > data.loc[["row_name"]]["col_name"]  # as a Series
> > data.loc[["row_name"]][["col_name"]]  # as a DataFrame
> > 
> > data.loc["row_name", "col_name"]  # as a value
> > data.loc[["row_name"], "col_name"]  # as a Series. Preserves index. Column name is moved to `.name`.
> > data.loc["row_name", ["col_name"]]  # as a Series. Index is moved to `.name.` Sets index to column name.
> > data.loc[["row_name"], ["col_name"]]  # as a DataFrame (preserves original index and column name)
> > 
> > # by column/row names: Dot notation
> > data.col_name.row_name
> > 
> > # by column/row indices
> > data.iloc[row_index, col_index] # as a value
> > data.iloc[[row_index], col_index] # as a Series. Preserves index. Column name is moved to `.name`
> > data.iloc[row_index, [col_index]] # as a Series. Index is moved to `.name.` Sets index to column name.
> > data.iloc[[row_index], [col_index]] # as a DataFrame (preserves original index and column name)
> > 
> > # column name + row index
> > data["col_name"][row_index]
> > data.col_name[row_index]
> > data["col_name"].iloc[row_index]
> > 
> > # column index + row name
> > data.iloc[:, [col_index]].loc["row_name"]  # as a Series
> > data.iloc[:, [col_index]].loc[["row_name"]]  # as a DataFrame
> > 
> > # using masks
> > data[data.index == "row_name"].T[data.T.index == "col_name"].T
> > ```
> > 
> > 4\. Access several columns:
> > 
> > ```python
> > # by name
> > data[["col1", "col2", "col3"]]
> > data.loc[:, ["col1", "col2", "col3"]]
> > 
> > # by index
> > data.iloc[:, [col1_index, col2_index, col3_index]]
> > ```
> > 
> > 5\. Access several rows
> > 
> > ```python
> > # by name
> > data.loc[["row1", "row2", "row3"]]
> > 
> > # by index
> > data.iloc[[row1_index, row2_index, row3_index]]
> > ```
> > 
> > 6\. Access a subset of specific rows and columns
> > 
> > ```python
> > # by names
> > data.loc[["row1", "row2", "row3"], ["col1", "col2", "col3"]]
> > 
> > # by indices
> > data.iloc[[row1_index, row2_index, row3_index], [col1_index, col2_index, col3_index]]
> > 
> > # column names + row indices
> > data[["col1", "col2", "col3"]].iloc[[row1_index, row2_index, row3_index]]
> > 
> > # column indices + row names
> > data.iloc[:, [col1_index, col2_index, col3_index]].loc[["row1", "row2", "row3"]]
> > ```
> > 
> > 7\. Access a subset of row and column ranges
> > 
> > ```python
> > # by name
> > data.loc["row1":"row2", "col1":"col2"]
> > 
> > # by index
> > data.iloc[row1_index:row2_index, col1_index:col2_index]
> > # column names + row indices
> > data.loc[:, "col1_name":"col2_name"].iloc[row1_index:row2_index]
> > 
> > # column indices + row names
> > data.iloc[:, col1_index:col2_index].loc["row1":"row2"]
> > ```
> { : .solution}
{: .challenge}

:::::::::::::::::::::::::::::::::::::::  challenge

## Exploring available methods using the `dir()` function

Python includes a `dir()` function that can be used to display all of the available methods (functions) that are built into a data object.  In Episode 4, we used some methods with a string. But we can see many more are available by using `dir()`:

```python
my_string = 'Hello world!'   # creation of a string object 
dir(my_string)
```

This command returns:

```python
['__add__',
...
'__subclasshook__',
'capitalize',
'casefold',
'center',
...
'upper',
'zfill']
```

You can use `help()` or <kbd>Shift</kbd>\+<kbd>Tab</kbd> to get more information about what these methods do.

Assume Pandas has been imported and the Gapminder GDP data for Europe has been loaded as `data`.  Then, use `dir()`
to find the function that prints out the median per-capita GDP across all European countries for each year that information is available.

:::::::::::::::  solution

## Solution

Among many choices, `dir()` lists the `median()` function as a possibility.  Thus,

```python
data.median()
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Interpretation

Poland's borders have been stable since 1945,
but changed several times in the years before then.
How would you handle this if you were creating a table of GDP per capita for Poland
for the entire twentieth century?


::::::::::::::::::::::::::::::::::::::::::::::::::

[pandas-dataframe]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
[pandas-series]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html
[numpy]: https://www.numpy.org/


