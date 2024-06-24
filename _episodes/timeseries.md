---
title: Time Series Analysis with Python
teaching: 1
exercises: 0
questions:
- "What are the primary advantages of using Recurrent Neural Networks (RNNs) for time series forecasting over traditional statistical methods and other machine learning algorithms?"
- "How do Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) enhance the capability of RNNs in learning and remembering temporal dependencies in sequential data?"
- "What recent advancements in RNN variants, such as the Temporal Fusion Transformer (TFT), have contributed to improved time series forecasting in physical sciences applications?"
objectives:
- "To identify the advantages of Recurrent Neural Networks (RNNs) in time series forecasting compared to traditional statistical methods and other machine learning algorithms."
- "To understand the role of Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) in enhancing the memory and temporal dependency learning capabilities of RNNs."
- "To explore the recent advancements in RNN variants, such as the Temporal Fusion Transformer (TFT), and their impact on time series forecasting in physical sciences."
keypoints:
- "The application of deep learning, particularly through RNNs and their variants like LSTM, GRU, and TFT, holds significant promise for time series forecasting in the physical sciences"
- "By harnessing the memory and pattern recognition capabilities of these networks, it is possible to achieve more accurate and insightful predictions on time-dependent data."
- "LSTMs and GRUs possess mechanisms to remember important events over extended periods."
- "LSTM, GRU and TFT models leverage advanced mechanisms for superior predictive performance in physical sciences applications."
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# Time Series Modeling 

Time series data consists of observations recorded at regular intervals, and time series forecasting aims to predict future values based on historical patterns and trends. This type of forecasting is critical across various domains, making it highly pertinent for machine learning practitioners. Applications such as risk scoring, fraud detection, and weather forecasting all benefit significantly from accurate time series predictions.

Deep learning, particularly neural networks, has only recently started to outperform traditional methods in time series forecasting, albeit by a smaller margin compared to its successes in image and language processing. Despite this, deep learning presents distinct advantages over other machine learning techniques, such as gradient-boosted trees. The primary advantage lies in the ability of neural network architectures to inherently understand time, automatically linking temporally close data points.

A Recurrent Neural Network (RNN) is a type of artificial neural network that includes memory or feedback loops, allowing it to recognize patterns in sequential data more effectively. The recurrent connections enable the network to consider both the current data sample and its previous hidden state, enhancing its ability to capture complex temporal dependencies. Some RNNs, like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), have specialized memory mechanisms to remember significant events over extended time periods, which is essential for sequence learning tasks.

The distinguishing feature of sequence learning, compared to other regression and classification tasks, is the necessity to utilize models that can learn temporal dependencies in input data. LSTMs and GRUs excel in this regard, offering the ability to capture intricate time dependencies. Furthermore, neural networks employ continuous activation functions, making them particularly adept at interpolation in high-dimensional spaces, thereby optimizing inputs effectively.

Recent advancements in RNN variants, such as the Temporal Fusion Transformer (TFT), have further enhanced the capabilities of time series forecasting in physical science applications. These state-of-the-art models leverage advanced attention mechanisms and complex architectural innovations to provide superior predictive performance.


## Time Series EDA

- Let's start with calculating the rolling mean and standard deviation, adds upper and lower bounds, and visualizes the \\( CO_2 \\)levels data trends over time in a plot with enhanced aesthetics using seaborn:

~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
co2_levels = pd.read_csv('data/co2_levels.csv')

# Convert the 'timestamp' column to datetime format
co2_levels['datestamp'] = pd.to_datetime(co2_levels['datestamp'])

# Set the 'timestamp' column as the index
co2_levels.set_index('datestamp', inplace=True)

# Compute the 52 weeks rolling mean of the co2_levels DataFrame
ma = co2_levels.rolling(window=52).mean()

# Compute the 52 weeks rolling standard deviation of the co2_levels DataFrame
mstd = co2_levels.rolling(window=52).std()

# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['co2'] + (mstd['co2'] * 2)

# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['co2'] - (mstd['co2'] * 2)

# Plot the content of the ma DataFrame
ax = ma.plot(linewidth=0.8, fontsize=6)

# Specify labels, legend, and show the plot
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel(r'$CO_2$ levels', fontsize=10)
ax.set_title(r'Rolling mean and variance of $CO_2$ levels\n from 1958 to 2001', fontsize=10)
plt.show();

~~~
{: .python}


![](../fig/co2_rolling_mean.png)


The increasing rolling mean of \\( CO_2 \\)  levels suggests a long-term upward trend in atmospheric \\( CO_2 \\) concentrations, which is a key indicator of climate change. This trend is likely driven by human activities and has important implications for the environment and global climate.



> ## Exercise: Plot the \\(CO_2\\) time series with a vertical line
> 
> - Plot the \\(CO_2\\) levels over time using the `plot()` method
> - Add a red vertical line at the date '1960-01-01' using `axvline()`
> - Set the x-axis label to 'Date' and the title to 'Number of Monthly CO2'
> 
> > ## Solution
> > ~~~
> > # Plot the time series in your dataframe
> > ax = co2_levels.plot(color='blue', fontsize=12)
> > # Add a red vertical line at the date 1960-01-01
> > ax.axvline('1960-01-01', color='red', linestyle='--')
> > # Specify the labels in your plot
> > ax.set_xlabel('Date', fontsize=12)
> > ax.set_title(r'Number of Monthly $CO_2$', fontsize=12)
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_tends.png)
> {: .solution}
> 
{: .challenge}



## Treand, Seasonality and Noise

In time series analysis, data often comprises three main components: seasonality, trend, and noise. Seasonality refers to recurring patterns at regular intervals, trend indicates long-term direction, and noise represents random fluctuations. Understanding these components is essential for accurate analysis and forecasting. The code snippet below demonstrates time series decomposition using `seasonal_decompose` from `statsmodels` to extract and visualize the trend component of the \\(CO_2\\) time-series data.

~~~
# Import statsmodels.api as sm
import statsmodels.api as sm
# Perform time series decompositon
decomposition = sm.tsa.seasonal_decompose(co2_levels)
# Extract the trend component
trend = decomposition.trend
# Plot the values of the trend
ax = trend.plot(figsize=(12, 6), fontsize=6)
# Specify axis labels
ax.set_xlabel('Date', fontsize=10)
ax.set_title(r'Trend component of  $CO_2$ time-series', fontsize=10)
plt.show()
~~~
{: .python}


![](../fig/co2_trends_component.png)


> ## Exercise 1: Plot the seasonal components of CO2 time series
> 
> - Import the `statsmodels.api` module as `sm`
> - Perform time series decomposition using `seasonal_decompose` on the `co2_levels` DataFrame
> - Extract the seasonal components from the decomposition
> - Plot the seasonal component values using `plot()` 
> - Set the x-axis label to 'Date' and the title to 'Seasonal component of CO2 time-series'
> 
> > ## Solution
> > ~~~
> > import statsmodels.api as sm
> > # Perform time series decomposition
> > decomposition = sm.tsa.seasonal_decompose(co2_levels)
> > # Extract the seasonal components
> > seasonal = decomposition.seasonal
> >
> > # Plot the values of the seasonality 
> > ax = seasonal.plot(figsize=(12, 6), fontsize=6)
> > # Specify axis labels
> > ax.set_xlabel('Date', fontsize=10)
> > ax.set_title(r'Seasonal component of $CO_2$ time-series', fontsize=10)
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_seasonal_component.png)
> {: .solution}
> 
{: .challenge}


> ## Exercise 2: Decompose and Plot \\(CO2\\) Time Series with Log Scale
> In this exercise, you will perform time series decomposition on CO2 levels data, extract the trend and seasonal components, and plot the decomposed time series using a log scale.
> - Decompose \\(CO_2\\) time series, extract the trend and seasonal components, and plot the decomposed time series using a log scale on the y-axis.
>
> > ## Solution
> > ~~~
> > from statsmodels.tsa.seasonal import seasonal_decompose
> > import numpy as np
> > import pandas as pd
> > import matplotlib.pyplot as plt
> > import seaborn as sns
> > plt.style.use('ggplot')
> > co2_levels = pd.read_csv('data/co2_levels.csv')
> > # Convert the 'timestamp' column to datetime format
> > co2_levels['datestamp'] = pd.to_datetime(co2_levels['datestamp'])
> > # Set the 'timestamp' column as the index
> > co2_levels.set_index('datestamp', inplace=True)
> > # Perform time series decomposition
> > result = seasonal_decompose(co2_levels, model='additive', period=12)
> > # Plot the values of the df_decomposed DataFrame
> > ax = result.plot(figsize=(12, 6), fontsize=15, logy=True)
> > # Specify axis labels
> > ax.set_xlabel('Year', fontsize=15)
> > plt.legend(fontsize=15)
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_seasonal_trends.png)
> {: .solution}
{: .challenge}
 

> ## Exercise 3: Decompose and Plot CO2 Time Series Components
> - Load CO2 levels data and set the 'Year' column as the index.
> -  Perform time series decomposition with an additive model and a period of 12.
> -  Plot the observed, trend, seasonal, and residual components on separate subplots.
> - Adjust subplot spacing and display the plot.
> 
> > ## Solution
> > ~~~
> > from statsmodels.tsa.seasonal import seasonal_decompose
> > import numpy as np
> > import pandas as pd
> > import matplotlib.pyplot as plt
> > import seaborn as sns
> > plt.style.use('ggplot')
> > co2_levels = pd.read_csv('data/co2_levels.csv')
> > # Convert the 'timestamp' column to datetime format
> > co2_levels['Year'] = pd.to_datetime(co2_levels['datestamp'])
> > # Set the 'Year' column as the index
> > co2_levels.set_index('Year', inplace=True)
> > # Perform time series decomposition
> > result = seasonal_decompose(co2_levels['co2'], model='additive', period=12)
> > # Plot the decomposed components on individual subplots
> > fig, axes = plt.subplots(4, 1, figsize=(12, 8))
> > result.observed.plot(ax=axes[0])
> > axes[0].set_title('Observed')
> > result.trend.plot(ax=axes[1])
> > axes[1].set_title('Trend')
> > result.seasonal.plot(ax=axes[2])
> > axes[2].set_title('Seasonal')
> > result.resid.plot(ax=axes[3])
> > axes[3].set_title('Residual')
> > plt.tight_layout()
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_seasonal_trends_noise.png)
> {: .solution}
> 
{: .challenge}


