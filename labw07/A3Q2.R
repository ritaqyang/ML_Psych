
library(glmnet) 
library(psych)
library(pls)
set.seed(42)

mydata = read.csv('mobilephone2_training.csv')
mydata_test = read.csv('mobilephone2_test.csv')
mydata.X=data.matrix(mydata[,colnames(mydata)!="dura24"])
mydata_test.X=data.matrix(mydata_test[,colnames(mydata_test)!="dura24"])


lambdas = 10^seq(from = -5,to = 0, length.out = 100)

lasso.cv = cv.glmnet(mydata.X, mydata$dura24,
                     alpha = 1, lambda = lambdas,
                     nfold = 10)
plot(lambdas[100:1],lasso.cv$cvm[100:1],xlab="lambda",ylab="MSE")

lasso.lam=lasso.cv$lambda.min
lasso.lam
lasso.fit= glmnet(mydata.X, mydata$dura24,
                  alpha = 1, lambda = lasso.lam)
plot(lasso.cv)

lasso.pred.y.tt = predict(lasso.fit,newx = mydata_test.X)
lasso.MSE = mean((lasso.pred.y.tt - mydata_test$dura24)^2)
lasso.MSE

lasso.coef=coef(lasso.fit)
lasso.coef

