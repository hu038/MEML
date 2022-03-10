##This is the simulation code for the comparing statistical model and machine learning methods in longitudinal data

## generate data from mixed model
rm(list=ls())
library(nlme)
library(tree)
library(randomForest)
library (gbm)
library(e1071)
library(neuralnet)
library(knitr)
library(REEMtree)

K=100
K1=50
n=5

btrue=c(0.5,1,1.2)
sds = 2
sd = 1

genlme=function(K,n,btrue,sds,sd){
  id = rep(1:K, each = n)
  #id=as.factor(id)
  time=rep(1:n, K)
  
  Keff = rep( rnorm(K, 0, sds), each = n)
  neff = rnorm(K*n, 0, sd)
  
  x = rep( runif(K, 0, 1), each = n)
  
  t = runif(K*n, 0, 1) 
  
  mu=btrue[1] + btrue[2]*x + btrue[3]*t
  y = mu + Keff + neff
  
  dat=data.frame(id,x,t,y,time,mu)
  dat
}



sim=1000

####################two-step prediction
##n=5

RMSE.nntanh=RMSE.svmkernel=RMSE.lm=RMSE.lme=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=matrix(0,sim,3)
RMSE1.nntanh=RMSE1.svmkernel=RMSE1.lm=RMSE1.lme=RMSE1.REEM=RMSE1.tree=RMSE1.bag=RMSE1.rf=RMSE1.boost=RMSE1.svm=RMSE1.nn=matrix(0,sim,3)

beta1T=matrix(0,sim,3)
beta2T=matrix(0,sim,3)

for(s in 1:sim)
{
  cat("sim = ", s, "\n")
  data=genlme(K,n,btrue,sds,sd)
 
  for(j in 3:5){
  subset1= (data$time %in% j) 
  test=data[subset1,]
  subset2= (data$time %in% c(1:j-2)) 
  train=data[subset2,]
  
  fit.lm=lm(y~x+t,train)
  fit.lme=lme(y~x+t,train,random=~1|id,
              method="ML")
  fit.REEM<-REEMtree(y~x+t,train,random=~1|id)
  fit.tree=tree::tree(y~x+t,train)
  fit.bag =randomForest(y~x+t,train,mtry=2, importance =TRUE)
  fit.rf =randomForest(y~x+t,train,mtry=1, importance =TRUE)
  fit.boost =gbm(y~x+t,train, distribution=
                   "gaussian",n.trees =5000,interaction.depth =4)
  fit.svm=svm(y~x+t,train, kernel="linear", scale=FALSE)
  fit.svmkernel=svm(y~x+t,train, kernel="polynomial")
  # nndata=model.matrix(~yi+xi+time,train)
  # nndata=as.data.frame(nndata)
  fit.nn=neuralnet(y~x+t,train, hidden=2,linear.output = TRUE,stepmax = 1e6,threshold = 0.1)
  fit.nntanh=neuralnet(y~x+t,train,act.fct = "tanh", hidden=2,linear.output = TRUE,stepmax = 1e7,threshold = 0.1)
  
  beta1T[s,j-2]=fit.lm$coefficients[2]
  beta2T[s,j-2]=fit.lm$coefficients[3]
  
  a.lm=predict(fit.lm,test)
  a.lme=predict(fit.lme,test)
  a.REEM=predict(fit.REEM, test,id=test$id, EstimateRandomEffects=TRUE)
  a.tree=predict(fit.tree, test)
  a.bag=predict(fit.bag, test)
  a.rf=predict(fit.rf, test)
  a.boost=predict(fit.boost, test,n.trees =2000)
  a.svm=predict(fit.svm, test)
  a.svmkernel=predict(fit.svmkernel, test)
  a.nn=compute(fit.nn,test)
  a.nntanh=compute(fit.nntanh,test)
  
  ni=length(test$y)
  
  RMSE1.lm[s,j-2]=sqrt(sum((test$mu-a.lm)^2)/ni)
  RMSE1.lme[s,j-2]=sqrt(sum((test$mu-a.lme)^2)/ni)
  RMSE1.REEM[s,j-2]=sqrt(sum((test$mu-a.REEM)^2)/ni)
  RMSE1.tree[s,j-2]=sqrt(sum((test$mu-a.tree)^2)/ni)
  RMSE1.bag[s,j-2]=sqrt(sum((test$mu-a.bag)^2)/ni)
  RMSE1.rf[s,j-2]=sqrt(sum((test$mu-a.rf)^2)/ni)
  RMSE1.boost[s,j-2]=sqrt(sum((test$mu-a.boost)^2)/ni)
  RMSE1.svm[s,j-2]=sqrt(sum((test$mu-a.svm)^2)/ni)
  RMSE1.svmkernel[s,j-2]=sqrt(sum((test$mu-a.svmkernel)^2)/ni)
  RMSE1.nn[s,j-2]=sqrt(sum((test$mu-a.nn$net.result)^2)/ni)
  RMSE1.nntanh[s,j-2]=sqrt(sum((test$mu-a.nntanh$net.result)^2)/ni)
  
  RMSE.lm[s,j-2]=sqrt(sum((test$y-a.lm)^2)/ni)
  RMSE.lme[s,j-2]=sqrt(sum((test$y-a.lme)^2)/ni)
  RMSE.REEM[s,j-2]=sqrt(sum((test$y-a.REEM)^2)/ni)
  RMSE.tree[s,j-2]=sqrt(sum((test$y-a.tree)^2)/ni)
  RMSE.bag[s,j-2]=sqrt(sum((test$y-a.bag)^2)/ni)
  RMSE.rf[s,j-2]=sqrt(sum((test$y-a.rf)^2)/ni)
  RMSE.boost[s,j-2]=sqrt(sum((test$y-a.boost)^2)/ni)
  RMSE.svm[s,j-2]=sqrt(sum((test$y-a.svm)^2)/ni)
  RMSE.svmkernel[s,j-2]=sqrt(sum((test$y-a.svmkernel)^2)/ni)
  RMSE.nn[s,j-2]=sqrt(sum((test$y-a.nn$net.result)^2)/ni)
  RMSE.nntanh[s,j-2]=sqrt(sum((test$y-a.nntanh$net.result)^2)/ni)
 }

}

save.image("data_p2_100_5.RData")







