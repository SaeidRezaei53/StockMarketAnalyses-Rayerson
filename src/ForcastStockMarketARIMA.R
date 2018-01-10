#######################################################################################################################
# Script name: ForcastStockMarketARIMA.R
# Porpouse   : This script is developed to forcast the S&P 500 Index over the 10 years
#              I have used the ARIMA model (Time Series) in R
# Data source: Data source could be off-line (marketPriceHistory.csv) or online SP&500
# R Package usagae:
# ggplot2
# forecast
# plotly
# ggfortify
# tseries
# gridExtra
# docstring
# here
#######################################################################################################################
#Developer         Date              Version                    Reason
#Saeid Rezaei     2017-12-20           0                      Initial Version
#Saeid Rezaei     2018-01-05           1                   Added ARIMA method to script
#######################################################################################################################

#By Milind Paradkar

#“Prediction is very difficult, especially about the future”. Many of you must have come across this famous quote by Neils Bohr, 
#a Danish physicist. Prediction is the theme of this blog post. In this post, we will cover the popular ARIMA forecasting model 
#to predict returns on a stock and demonstrate a step-by-step process of ARIMA modeling using R programming.

#What is a forecasting model in Time Series?
#Forecasting involves predicting values for a variable using its historical data points or it can also 
#involve predicting the change in one variable given the change in the value of another variable. 
#Forecasting approaches are primarily categorized into qualitative forecasting and quantitative forecasting. 
#Time series forecasting falls under the category of quantitative forecasting wherein statistical principals and concepts are 
#applied to a given historical data of a variable to forecast the future values of the same variable. 
#Some time series forecasting techniques used include:

#Autoregressive Models (AR)
#Moving Average Models (MA)
#Seasonal Regression Models
#Distributed Lags Models

#What is Autoregressive Integrated Moving Average (ARIMA)?
#ARIMA stands for Autoregressive Integrated Moving Average. ARIMA is also known as Box-Jenkins approach. 
#Box and Jenkins claimed that non-stationary data can be made stationary by differencing the series, 
#Yt. The general model for Yt is written as,

#Yt =?1Yt-1 +?2Yt-2…?pYt-p +?t + ?1?t-1+ ?2?t-2 +…?q?t-q

#Where, Yt is the differenced time series value, ? and ? are unknown parameters and ? are independent 
#identically distributed error terms with zero mean. Here, Yt is expressed in terms of its past values and the 
#current and past values of error terms.

print ("STEP 1.0: Installing the packages...")
install.packages("ggplot2")
install.packages("forecast")
install.packages("plotly")
install.packages("ggfortify")
install.packages("tseries")
install.packages("gridExtra")
install.packages("docstring")
install.packages("here")

library(ggplot2)
library(forecast)
library(plotly)
library(ggfortify)
library(tseries)
library(gridExtra)
library(docstring)
library(here)

# Find the local / Working directory and copy all the project there
print ("STEP 1.1: Find the working directory.")

here()
source(here("src",'main_functions.R'))
# NOTE: For more information on helper functions use ?function_name

print ("STEP 1.2: Loading Data.")

# LOAD DATA
dataMaster <- read.csv(here("data", "SP500Data.csv"))
attach(dataMaster)
print ("STEP 2.1: Start Analysing.")
# EXPLORATORY ANALYSIS
# I'm going to get valumn from 1995 to present with frq 12
sp_500 <- ts(dataMaster$sp_500, start=c(1995, 1), freq=12)

# TESTS FOR STATIONARITY
Box.test(sp_500, lag = 20, type = 'Ljung-Box')
adf.test(sp_500)

# p-values are relatively high so we should so visual inspection and
# look at ACF and PACF plots to make appropriate transformation 
# for stationarity. 



# TIME SERIES PLOT OF S&P
tsSp <- plot_time_series(sp_500, 'S&P 500')

tsSp
ggplotly(tsSp)

# Here we create the training set where we will compare the values for 2015 
sp500_TR <- ts(sp_500, start=c(1995, 1), end=c(2014, 12), freq=12)
plot_time_series(sp_500, 'S&P 500 Training Set')

print ("STEP 2.2: plotting the SP500")
# DECOMPOSING TIME SERIES
sp500_stl <- plot_decomp(sp500_TR, 'S&P 500')
sp500_stl
ggplotly(sp500_stl)


# SEASONAL PLOT 
seasonal_Plot <- plot_seasonal(sp500_TR, 'S&P 500')

seasonal_Plot
ggplotly(seasonal_Plot)


# DIAGNOSING ACF AND PACF PLOTS
plot_acf_pacf(sp500_TR, 'S&P 500')
# TRANSFORMING OUR DATA TO ADJUST FOR NON STATIONARY
diff <- diff(sp_500)

tsDiff <- plot_time_series(diff, 'First Difference')
tsDiff
ggplotly(tsDiff)


# TESTS FOR STATIONARITY FOR DIFFERENCED TIME SERIES OBJECT
Box.test(diff, lag = 20, type = 'Ljung-Box')
adf.test(diff)


# p-values seems small enough to infer stationarity for the first difference
# Let's begin analysis with visually inspecting ACF and PACF plots

# DIAGNOSING ACF AND PACF PLOTS FOR DIFFERENCED TIME SERIES OBJECT
plot_acf_pacf(diff, 'First Difference Time Series Object')

# SEASONAL PLOT FOR DIFFERENCED TIME SERIES OBJECT
spDiff <- plot_seasonal(diff, 'First Difference Time Series Object')
spDiff
ggplotly(spDiff)


# AUTO.ARIMA ESTIMATION
auto.arima(sp500_TR)

# From our visual inspection and auto.arima model we will choose an
# ARIMA(0, 1, 1) with drift 

# BUILD MODEL 
fit <- Arima(sp500_TR, order = c(0,1,1), include.drift = TRUE)
summary(fit)


# RESIDUAL DIAGNOSTICS
ggtsdiag(fit) + 
  theme(panel.background = element_rect(fill = "gray98"),
        panel.grid.minor = element_blank(),
        axis.line.y = element_line(colour="gray"),
        axis.line.x = element_line(colour="gray")) 

residFit <- ggplot(data=fit, aes(residuals(fit))) + 
  geom_histogram(aes(y =..density..),  
                 binwidth = 5,
                 col="turquoise4", fill="white") +
  geom_density(col=1) +
  theme(panel.background = element_rect(fill = "gray98"),
        panel.grid.minor = element_blank(),
        axis.line   = element_line(colour="gray"),
        axis.line.x = element_line(colour="gray")) +
  ggtitle("Plot of SP 500 ARIMA Model Residuals") 

residFit


# TEST SET THAT WE WILL COMPARE OUR FORECAST AGAINST 
dataMaster_TS <- dataMaster[-c(1:240), ]
act_sp500_2015_ts <- ts(dataMaster_TS$sp_500, start = c(2015, 1), freq = 12)
act_sp500_2015_ts


print ("STEP 2.2: Forcasting.")

# FORECASTING
# METHOD CHOSEN THROUGH BOX JENKINS METHODOLOGY WAS ARIMA(0,1,1) WITH DRIFT
## ARIMA MODEL CHOSEN 
fit_arima <- forecast(fit, h = 12)
forSp500 <- autoplot(fit_arima, 
                     holdout = act_sp500_2015_ts, 
                     ts_object_name = 'ARIMA')

forSp500
ggplotly(forSp500)

## BOX COX TRANSFORMATION
lambda <- BoxCox.lambda(sp500_TR)
fit_sp500_BC <- ar(BoxCox(sp500_TR,lambda))
fit_BC <- forecast(fit_sp500_BC,h=12,lambda=lambda)

s <- autoplot(fit_BC, 
              holdout = act_sp500_2015_ts,
              ts_object_name = 'Box-Cox Transformation')
s
ggplotly(s)

# MEAN FORECAST METHOD
fit_meanf <- forecast(meanf(sp500_TR, h = 12))
e <- autoplot(fit_meanf, 
              holdout = act_sp500_2015_ts,
              ts_object_name = 'Mean Forecast') 
e
ggplotly(e)

# NAIVE METHOD
fit_naive <- forecast(naive(sp500_TR, h = 12))
f <- autoplot(fit_naive, 
              holdout = act_sp500_2015_ts,
              ts_object_name = "Naive Forecast") 
f
ggplotly(f)

# SEASONAL NAIVE METHOD
fit_snaive <- forecast(snaive(sp500_TR, h = 12))
g <- autoplot(fit_snaive, 
              holdout = act_sp500_2015_ts,
              ts_object_name = "Seasonal Naive")
g
ggplotly(g)  

# EXPONENTIAL SMOOTHING METHOD
fit_ets <- forecast(ets(sp500_TR), h = 12)
h <- autoplot(fit_ets, 
              holdout=act_sp500_2015_ts,
              ts_object_name = "Exponential Smoothing")

h
ggplotly(h)  

# COMPARE FORECAST ACCURACIES ACROSS DIFFERENT METHODS USED
accuracy(fit_arima)
accuracy(fit_BC)
accuracy(fit_meanf)
accuracy(fit_naive)
accuracy(fit_snaive)
accuracy(fit_ets)

# CONCLUSIONS
# The model with the best diagnostics is our ARIMA Model 

# ARCH Modeling
# Here we first square the residuals and plot the time series/ACF/PACF 
# to see if there is correlation within the residuals.
# If there is we can continue adding on to our ARIMA model with a gARCH 
# aspect that helps in the volatity of our data.
squared_res_fit <- fit$residuals^2

sq_res <- plot_time_series(squared_res_fit, "Squared Residuals")

sq_res
ggplotly(sq_res)

# ACF AND PACF PLOT FOR SQUARED RESIDUALS 
plot_acf_pacf(squared_res_fit, 'S&P 500 Residuals^2')
# The acf plot shows one significant lag, as does the pacf, 
# but that isn't enough to suggest we need GARCH modeling
gfit <- garch(fit$residuals, order = c(1,1), trace = TRUE)


print ("end of script.")