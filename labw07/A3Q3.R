library(glmnet) 
library(psych)
library(pls)
set.seed(42)

mydata = read.csv('mobilephone2_training.csv')
mydata_test = read.csv('mobilephone2_test.csv')
mydata.X=data.matrix(mydata[,colnames(mydata)!="dura24"])
mydata_test.X=data.matrix(mydata_test[,colnames(mydata_test)!="dura24"])