#transfering file to main 

mydata = read.csv("druguse_training.csv")

mydata$gender = factor(mydata$gdender, levels = c(0,1), labels = c('male', 'female') )
mydata$family = factor(mydata$family, levels = c(2,1), labels = c('others', 'step or foster families'))
mydata$marital = factor(mydata$marital, levels = c(0,1), labels = c('single', 'married'))

lm_fit_long = lm(cig1 ~ gender + age + ses + marital + family, mydata) 
summary(lm_fit_long)

lm_fit_long = lm(cig1 ~ gender + age + I(age^2) + ses + marital + family, mydata) 
summary(lm_fit_long)

# 3. KNN regression ----
## Run a model ----
install.packages("caret")
library(caret)
mydata = read.csv("druguse_training.csv")
mydata_test = read.csv("druguse_test.csv") 
knnmodel = knnreg(cig1 ~ gender + age + ses + marital + family, mydata, k = 5)

## Calculate R2 ----
pred_y_knn = predict(knnmodel, mydata)
R2_knn = 1 - sum((mydata$cig1 - pred_y_knn)^2)/sum((mydata$cig1 - mean(mydata$cig1))^2)

lm.fit1 = lm(cig1 ~ gender + age + ses + marital + family, mydata) # multiple linear regression
R2_lm1 = 1 - sum(lm.fit1$residuals^2)/sum((mydata$cig1 - mean(mydata$cig1))^2)

cat("\n R2_knn: ", R2_knn, "\n R2_lm1: ", R2_lm1)

# 4. Model Comparison (MSE) ----
## Run competing models ----
lm.fit1 = lm(cig1 ~ gender + age + ses + marital + family, mydata)  # multiple linear regression
lm.fit2 =lm(cig1 ~ gender + age + I(age^2) + ses + marital + family, mydata) # Polynomial regression w/ quadratic term 

## Calculate MSE values ---- 
pred_y_knn = predict(knnmodel, mydata_test)
pred_y_lm1 = predict(lm.fit1, mydata_test)
pred_y_lm2 = predict(lm.fit2, mydata_test)

MSE_knn = mean((mydata_test$cig1 - pred_y_knn)^2)
MSE_lm1 = mean((mydata_test$cig1 - pred_y_lm1)^2)
MSE_lm2 = mean((mydata_test$cig1 - pred_y_lm2)^2)


cat("\n MSE_knn: ", MSE_knn, "\n MSE_lm1: ", MSE_lm1,
    "\n MSE_lm2: ", MSE_lm2)

cat("\n\n\n\n\n MSE_knn: ", MSE_knn)

