#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week #2: Regression 1                 #
# Instructor: Heungsun Hwang            #
# TA: Gyeongcheol Cho                   #
#===#===#===#===#===#===#===#===#===#===#

#install.packages("psych")  # for calculating descriptive statistics 

# 0. Setting the working directory
  Susetwd('C:\\Users\\cheol\\Dropbox\\Mcgill\\Lecture\\Machine_Learning_Hwang\\2023_Winter\\W02_Regression 1')
    # Note that "/" or "\\" should be used for path separator, not "\"  

# 1. Simple linear regression (SLR) ----
  mydata = read.csv("wineheartattack.csv") 
  View(mydata)
  summary(mydata)
  
  describe(mydata)
  library(psych)
  describe(mydata)
  
  lm_fit = lm(heartattack ~ wine, mydata) 
  summary(lm_fit)
  
  plot(mydata$wine, mydata$heartattack)
  abline(lm_fit, lwd = 3, col = "red")

# 2. Multiple linear regression (MLR) ----
  ## Run a model ----
  mydata = read.csv("curran_training.csv") 
  View(mydata)
  
  summary(mydata)
  describe(mydata)
  
  lm_fit_long = lm(anti1 ~ gender + cogstm + emotsup, mydata)
  summary(lm_fit_long)
  
  lm_fit_short = lm(anti1 ~ gender+emotsup, mydata)
  summary(lm_fit_short)
  
  ## Calculate predicted values/MSE/RMSE ----
  mydata_test = read.csv("curran_test.csv") 
  pred_y_test_short = predict(lm_fit_short, mydata_test)
  MSE_short = mean((mydata_test$anti1 - pred_y_test_short)^2)
  RMSE_short = sqrt(MSE_short)

  pred_y_test_long = predict(lm_fit_long, mydata_test)
  MSE_long = mean((mydata_test$anti1 - pred_y_test_long)^2)
  RMSE_long = sqrt(MSE_long)
  
  cat("\n MSE_long : ", MSE_long,
      "\n MSE_short: ", MSE_short)