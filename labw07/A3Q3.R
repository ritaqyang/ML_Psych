
#A3Q3 

#Principle component regression (PCR)

library(glmnet) 
library(psych)
library(pls)
set.seed(42)

mydata = read.csv('mobilephone2_training.csv')
mydata_test = read.csv('mobilephone2_test.csv')
mydata.X=data.matrix(mydata[,colnames(mydata)!="dura24"])
mydata_test.X=data.matrix(mydata_test[,colnames(mydata_test)!="dura24"])


# 3. Data Reduction - PCA ----

mydata2 = read.csv('mobilephone2_training.csv')
mydata2[,1] = NULL
head (mydata2)
scree(mydata2,factors=FALSE)

pca.fit1 = principal(mydata2, nfactors = 2,rotate="none")
pca.fit2 = principal(mydata2, nfactors = 2,rotate="varimax")
# "none", "varimax", "quartimax", "promax", "oblimin", "simplimax", and "cluster" 
pca.fit1
pca.fit2

pca.fit2$loadings
pca.fit2$weights
pca.fit2$scores


pcr.fit = pcr(dura24 ~ .,data=mydata,
              scale = TRUE, validation = "CV")
summary(pcr.fit)

validationplot(pcr.fit) 
pcr.pred.y.tt = predict(pcr.fit, mydata_test, ncomp = 9)
pcr.MSE =mean((pcr.pred.y.tt - mydata_test$dura24)^2) 
pcr.MSE
#head(pcr.fit$scores)

## 4-2. PLSR ----
plsr.fit = plsr(dura24 ~ ., data = mydata,
                scale = TRUE, validation = "CV")
summary(plsr.fit)
validationplot(plsr.fit) 
plsr.pred.y.tt = predict(plsr.fit, mydata_test, ncomp = 2)

## 4-3. Comparison ----
pcr.MSE =mean((pcr.pred.y.tt - mydata_test$dura24)^2) 
plsr.MSE=mean((plsr.pred.y.tt - mydata_test$dura24)^2)
cat("\n MSE_PCR:  ",pcr.MSE,
    "\n MSE_PLSR: ",plsr.MSE)
