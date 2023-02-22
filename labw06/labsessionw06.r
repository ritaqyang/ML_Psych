#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week #6: Resampling Methods           #
# Heungsun Hwang & Gyeongcheol Cho      #
#===#===#===#===#===#===#===#===#===#===#

# 1. Preparation ----
  setwd('C:/Users/cheol/Dropbox/Mcgill/Lecture/Machine_Learning_Hwang/2023_Winter/W06_Resampling_Methods')
  mydata=read.csv("auto_mpg.csv")
  View(mydata)
  
# 2. Cross Validation ----
# 2-1. Validation Set Approach ----
  ID = sample(392,392)
  mydata.reordered = mydata[ID,]
  View(mydata.reordered)
  
  mydata.training = mydata.reordered[1:196,] 
  mydata.validation = mydata.reordered[197:392,]
  
  plm.fit1 = lm(mpg ~ horsepower,data=mydata.training)
  plm.fit2 = lm(mpg ~ poly(horsepower,2),data=mydata.training)
  plm.fit3 = lm(mpg ~ poly(horsepower,3),data=mydata.training)
  
  mse1=mean((mydata.validation$mpg - predict(plm.fit1, mydata.validation))^2)
  mse2=mean((mydata.validation$mpg - predict(plm.fit2, mydata.validation))^2)
  mse3=mean((mydata.validation$mpg - predict(plm.fit3, mydata.validation))^2)
  c(mse1,mse2,mse3)
  
  List.poly=1:10
  list.mse.vsa=rep(0,10)
  for (i in List.poly){
      plm.fit.vsa = lm(mpg ~ poly(horsepower,i),data=mydata.training)
      list.mse.vsa[i] = mean((mydata.validation$mpg - predict(plm.fit.vsa, mydata.validation))^2)
  }
  plot(List.poly,list.mse.vsa, type="b",main="Validaiton set approach",xlab="degree of polynomial",ylab="MSE",col="red")
  
# 2-2. LOOCV Approach ----
  list.mse.loocv=rep(0,10)
  for (i in List.poly){
    plm.fit.loocv = glm(mpg ~ poly(horsepower,i),data=mydata)
    list.mse.loocv[i] = cv.glm(mydata,plm.fit.loocv)$delta[1]
  }
  plot(List.poly,list.mse.loocv,type="b",main="LOOCV",xlab="degree of polynomial",ylab="MSE",col="red")

# 2-3. K-fold Approach ----
  list.mse.loocv=rep(0,10)
  for (i in List.poly){
    plm.fit.loocv = glm(mpg ~ poly(horsepower,i),data=mydata)
    list.mse.loocv[i] = cv.glm(mydata,plm.fit.loocv,K=10)$delta[1]
  }
  plot(List.poly,list.mse.loocv,type="b",main="K-fold approach",xlab="degree of polynomial",ylab="MSE",col="red")

  plm.fit=lm(mpg ~ poly(horsepower,2),data=mydata)
  summary(plm.fit)

# 2-4. K-fold Cross validation that can be used for general purpose  ----
  # 2-4-1. specify a set of hyperparameter values considered and choose K for K-fold CV 
  list.para=1:10
  K=10
  # 2-4-2. specify how to fit the model and to calculate validation error
  genCVE = function(mydata,i,loc.test){
    mydata.test=mydata[loc.test,]
    mydata.train=mydata[-loc.test,]
    rm(mydata)
    ####################### Users need to specify ############################
    model.fit = lm(mpg ~ poly(horsepower,i),data=mydata.train) # for training 
    mean((mydata.test$mpg - predict(model.fit, mydata.test))^2) # for testing
    ##########################################################################
  }
  # 2-4-3. load the CVE function 
  load("CVE.RData")
  # 2-4-4. Run K-CV
  Result=CVE(mydata,genCVE,list.para,K)
  
# 3. Bootstrapping method ----
  #install.packages("boot")  # for the bootstrapping method
  library(boot)
  
  # 3-1. Train the model
  model.fit = lm(mpg ~ horsepower+I(horsepower^2),data=mydata)
  summary(model.fit)
  
  # 3-2. Specify how to obtain parameter estimates of interest
  genEst = function(mydata,loc.train){
    mydata.train=mydata[loc.train,]
    rm(mydata)
    ####################### Users need to specify ############################
    model.fit = lm(mpg ~ horsepower+I(horsepower^2),data=mydata.train) # for training 
    coef(model.fit)
    ##########################################################################
  }
  SE.boot=boot(mydata,genEst,1000)
  boot.ci(SE.boot,type=c("basic"),index=1)
  boot.ci(SE.boot,type=c("basic"),index=2)
  boot.ci(SE.boot,type=c("basic"),index=3)
  
  
