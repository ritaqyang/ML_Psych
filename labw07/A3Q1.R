
library(glmnet) 
library(psych)
library(pls)
set.seed(42)
## 2-2. While searching for the best lambda ----
### 2-2-1. Ridge regression ----

mydata = read.csv('mobilephone2_training.csv')
mydata_test = read.csv('mobilephone2_test.csv')
mydata.X=data.matrix(mydata[,colnames(mydata)!="dura24"])
mydata_test.X=data.matrix(mydata_test[,colnames(mydata_test)!="dura24"])


lambdas = 10^seq(from = -5,to = 0, length.out = 100)
#length.out is the total length of the sequence 
cat(lambdas)

ridge.cv = cv.glmnet(mydata.X, mydata$dura24,
                     alpha = 0, lambda = lambdas,
                     nfold = 10)
plot(lambdas[100:1],ridge.cv$cvm[100:1],xlab="lambda",ylab="MSE")

ridge.lam=ridge.cv$lambda.min
ridge.lam
ridge.fit= glmnet(mydata.X, mydata$dura24,
                  alpha = 0, lambda = ridge.lam)
plot(ridge.cv)

## 2-3. Performance Evaluation ----
# Ridge
ridge.pred.y.tt = predict(ridge.fit, newx = mydata_test.X)
ridge.MSE = mean((ridge.pred.y.tt - mydata_test$dura24)^2)
ridge.MSE

##2-1. Given lambda ----
###2-1-1. Ridge regression ----

ridge.coef=coef(ridge.fit)
ridge.coef


