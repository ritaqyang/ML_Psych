#===#===#===#===#===#===#===#===#===#===#
# PSYC 560                              #
# Week 10: Cluster Analysis             #
# Heungsun Hwang and Gyeongcheol Cho    #
#===#===#===#===#===#===#===#===#===#===#
#install.packages('mclust') # 

# 0. Load libraries & data ---- 
library(mclust)
setwd('C:/Users/cheol/Dropbox/Mcgill/Lecture/Machine_Learning_Hwang/2023_Winter/W10_Cluster_Analysis')

mydata = read.csv('shoppingattitude.csv')
head(mydata)
mydata = mydata[,-1]
head(mydata)

mydata.bases = mydata[,c(1,2,3,4,5,6)]
mydata.descriptors = mydata[,c(7,8)]
pairs(mydata.bases,upper.panel=NULL)
# refer to http://www.sthda.com/english/wiki/scatter-plot-matrices-r-base-graphs

# 1. Cluster analysis
##1-1. Hierarchical Clustering ----
# mydata = scale(mydata) # standardization 
mydata.dist = dist(mydata.bases)
# computes and returns the Euclidean distance matrix
# method = "manhattan" - city-block distance

hc.fit = hclust(mydata.dist, method = "ward.D2") 
# method = "single","complete","average","centroid"
plot(hc.fit) 
height.cut = 3
group.hc = cutree(hc.fit,height.cut)
group.hc
pairs(mydata.bases,col = group.hc,pch = group.hc,upper.panel=NULL)

## 1-2. K-means  ----
# mydata = scale(mydata) # standardization 
K = 3
km.fit = kmeans(mydata.bases,K,nstart = 100)
km.fit
km.fit$tot.withinss # A vector of within-cluster sum of squares
km.fit$centers # A vector of cluster centers
km.fit$cluster # A vector of cluster membership
km.fit$size # # of members in each cluster
group.Km = km.fit$cluster
pairs(mydata.bases,col = group.Km,pch = group.Km,upper.panel=NULL)

# 1-3. Finite Mixture Models ----
fmm.BIC = mclustBIC(mydata.bases,modelNames="EII")
#modelNames="EII"
plot(fmm.BIC)
summary(fmm.BIC)

fmm.fit = Mclust(mydata.bases, x = fmm.BIC)
fmm.fit$z # probability 
fmm.fit$classification

pairs(mydata,col = fmm.fit$classification,
      pch = fmm.fit$classification,upper.panel=NULL)


# 2. Profiling analysis ----
Results.ind = aggregate(x   = mydata.descriptors, 
                        by  = list(group.Km), 
                        FUN = mean)
Results.ind
Results.ind = Results.ind[,-1] 
Results.ind
Results.ent = colMeans(mydata.descriptors)
Results.ent
Results = rbind(Results.ind, Results.ent)
Results
Results = t(Results)
Results
colnames(Results)=c("Group 1", "Group 2", "Group 3","Entire")
Results
write.csv(Results,'Profile.analysis.csv')

