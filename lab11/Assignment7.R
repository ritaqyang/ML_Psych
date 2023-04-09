library(torch)
library(luz) 
library(glmnet)
library(randomForest)
library(gbm)
# ########################################################################## #
# 1. Regression Problem ----
## 1.1. Data preprocessing  ----

mydata = read.csv('mobilephone3_training.csv')
head(mydata)

mydata.tt = read.csv('mobilephone3_test.csv')

N=nrow(mydata)
N.va=round(N*.2) #use 20% as validation data 
N.tr=N-N.va  #use the rest as training data
N.tt=nrow(mydata.tt)

id.vali = sample(1:N,N.va) 
mydata.va = mydata[id.vali,]
mydata.tr = mydata[-id.vali,]
#mydata.tt = mydata.tt

mydata.tr.X=model.matrix(dura24~.-1,data=mydata.tr)
mydata.tr.Y=mydata.tr$dura24
mydata.va.X=model.matrix(dura24~.-1,data=mydata.va)
mydata.va.Y=mydata.va$dura24
mydata.tt.X=model.matrix(dura24~.-1,data=mydata.tt)
mydata.tt.Y=mydata.tt$dura24

mean.tr.X=colMeans(mydata.tr.X)
sd.tr.X=apply(mydata.tr.X,2,sd) #2 indcates col and 1 indicate row 
#apply() function 
mydata.tr.X=(mydata.tr.X-matrix(1,N.tr,1)%*%t(mean.tr.X))/(matrix(1,N.tr,1)%*%sd.tr.X)
mydata.va.X=(mydata.va.X-matrix(1,N.va,1)%*%t(mean.tr.X))/(matrix(1,N.va,1)%*%sd.tr.X)
mydata.tt.X=(mydata.tt.X-matrix(1,N.tt,1)%*%t(mean.tr.X))/(matrix(1,N.tt,1)%*%sd.tr.X)

mydata.tr.X.torch=torch_tensor(mydata.tr.X,torch_float())
mydata.tr.Y.torch=torch_tensor(mydata.tr.Y,torch_float())
mydata.va.X.torch=torch_tensor(mydata.va.X,torch_float())
mydata.va.Y.torch=torch_tensor(mydata.va.Y,torch_float())

data.tr = list(mydata.tr.X.torch, mydata.tr.Y.torch)
data.va = list(mydata.va.X.torch, mydata.va.Y.torch)


## 1.2. Deep feedforward neural networks ----
### 1.2.1. Specifying a neural network ----
DL.R = nn_module(
  initialize = function() {
    self$hidden   = nn_linear(12, 80) #number of hidden units = 80 
    self$act.relu = nn_relu()
    self$dropout  = nn_dropout(0.5) #dropout rate given 
    self$output   = nn_linear(80, 1) 
    #self$act.sigmoid = nn_sigmoid() for a binary  classification problem
    #self$act.softmax = nn_log_softmax() for a classification problem with multiple classes
  },
  forward = function(input) {
    input %>%
      self$hidden() %>%
      self$act.relu() %>%
      self$dropout()  %>%
      self$output()
  }
)%>%
  setup(
    loss = nn_mse_loss(), 
    # nn_bce_loss() for a binary  classification problem
    # nn_nll_loss() for a classification problems with multiple classes
    optimizer = optim_rmsprop, # advanced SGD
    #metrics = list(luz_metric_mae())
    # luz_metric_mae(): MAE
    # luz_metric_rmse(): RMSE
  )%>%
  set_opt_hparams(weight_decay = 0.1)
#weight decay - ridge penalty parameter 

DL.R()


