#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week #3: Regression 2                 #
# Instructor: Heungsun Hwang            #
# TA: Gyeongcheol Cho                   #
#===#===#===#===#===#===#===#===#===#===#

#install.packages("psych")  # for calculating descriptive statistics 

# 0. Setting the working directory
  setwd('C:\\Users\\cheol\\Dropbox\\Mcgill\\Lecture\\Machine_Learning_Hwang\\W03_Regression 2')
    # Note that "/" or "\\" should be used for path separator, not "\"  

# 1. MLR with dummy variables ----
  mydata = read.csv("GlastonburyDummy.csv") 
  View(mydata)
  mydata$music = factor(mydata$music, levels = c(4,1,2,3), labels = c('others','indiekid','metaller','crusty')) 
    # "others" will be used as a baseline in regression. 
  View(mydata)
  summary(mydata)
  
  lm.fit = lm(change ~ music, mydata)
  summary(lm.fit)
  
# 2. Polynomial regression ----
  mydata = read.csv("curran_training.csv")
  View(mydata)
  describe(mydata)
  lm.fit = lm(anti1 ~ gender+cogstm+emotsup+I(emotsup^2), mydata)
  summary(lm.fit)

# 3. KNN regression ----
  ## Run a model ----
  #install.packages("caret")
  library(caret)
  mydata = read.csv("curran_training.csv")
  mydata_test = read.csv("curran_test.csv") 
  knnmodel = knnreg(anti1 ~ gender+cogstm+emotsup, mydata, k = 5)
  
  ## Calculate R2 ----
  pred_y_knn = predict(knnmodel, mydata)
  R2_knn = 1 - sum((mydata$anti1 - pred_y_knn)^2)/sum((mydata$anti1 - mean(mydata$anti1))^2)

  lm.fit1 = lm(anti1 ~ gender+cogstm+emotsup, mydata) # multiple linear regression
  R2_lm1 = 1 - sum(lm.fit1$residuals^2)/sum((mydata$anti1 - mean(mydata$anti1))^2)
  
  cat("\n R2_knn: ", R2_knn, "\n R2_lm1: ", R2_lm1)
  
# 4. Model Comparison (MSE) ----
  ## Run competing models ----
  lm.fit1 = lm(anti1 ~ gender+cogstm+emotsup, mydata) # multiple linear regression
  lm.fit2 = lm(anti1 ~ gender+cogstm+emotsup+I(cogstm^2), mydata) # Polynomial regression w/ quadratic term 
  lm.fit3 = lm(anti1 ~ gender+cogstm*emotsup+I(cogstm^2), mydata) # Polynomial regression w/ interaction & quadratic term 

  ## Calculate MSE values ---- 
  pred_y_knn = predict(knnmodel, mydata_test)
  pred_y_lm1 = predict(lm.fit1, mydata_test)
  pred_y_lm2 = predict(lm.fit2, mydata_test)
  pred_y_lm3 = predict(lm.fit3, mydata_test)

  MSE_knn = mean((mydata_test$anti1 - pred_y_knn)^2)
  MSE_lm1 = mean((mydata_test$anti1 - pred_y_lm1)^2)
  MSE_lm2 = mean((mydata_test$anti1 - pred_y_lm2)^2)
  MSE_lm3 = mean((mydata_test$anti1 - pred_y_lm3)^2)

  cat("\n MSE_knn: ", MSE_knn, "\n MSE_lm1: ", MSE_lm1,
      "\n MSE_lm2: ", MSE_lm2, "\n MSE_lm3: ", MSE_lm3,
      "\n MSE_lm4: ", MSE_lm4)

  cat("\n\n\n\n\n MSE_knn: ", MSE_knn)
  