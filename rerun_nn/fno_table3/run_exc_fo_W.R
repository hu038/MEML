##This is the simulation code for the comparing statistical model 
#and machine learning methods in longitudinal data

library(nlme)
library(tree)
library(randomForest)
library (gbm)
library(e1071)
library(neuralnet)
library(knitr)
library(REEMtree)

#################   EXC correlation structure
exc<- function(r,n)
{
 out <- diag(n)*.5
 out[row(out)!=col(out)]<-r*.5
 out+t(out)
}

########## correlated data generation
# y:  normal;
# x_i: uniform+c(1:n); random x.
# err_i: norm

gennorm<-function(K,n,b,cv) 
{
  y<-matrix(NA,K*n,7)
  dimnames(y)<-list(NULL,c("id","yi","xi","xi2","mu","time","err"))
  for (i in 1:K) {
    xi<- runif(n,-1,1)
    #+c(1,n) 
    id<-rep(i,n)
    time=1:n
    xi2=rbinom(1,1,0.5)
    mu<-b[1]+b[2]*xi+b[3]*xi2+b[4]*xi^2
    err<-t(cv)%*%rnorm(n)         #cv<-chol(cv), upper triangula	
    yi<-mu+sigma*err
    y[((i-1)*n+1):(i*n),] <-cbind(id,yi,xi,xi2,mu,time,err)
  }
  y
} 
 

####generate norm data
K=100
n=10
rho=0.5
sigma=1
btrue=c(0,0.5,1,-5)


cv=chol(exc(rho,n)) 


sim=1000

####################future observations

RMSE1.svmkernel=RMSE1.lm=RMSE1.lme=RMSE1.REEM=RMSE1.tree=RMSE1.bag=RMSE1.rf=RMSE1.boost=RMSE1.svm=RMSE1.nn=rep(0,sim)
RMSE.svmkernel=RMSE.lm=RMSE.lme=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=rep(0,sim)
BIAS.svmkernel=BIAS.lm=BIAS.lme=BIAS.REEM=BIAS.tree=BIAS.bag=BIAS.rf=BIAS.boost=BIAS.svm=BIAS.nn=rep(0,sim)

for(s in 1:sim)
{
  cat("sim = ", s, "\n")
  data=as.data.frame(gennorm(K,n,btrue,cv))
 
  subset2= (data$time %in% c(1:7)) 
  train=data[subset2,]
  test=data[!subset2,]
  
  fit.lm=lm(yi~xi+xi2,train)
    fit.lme=lme(yi~xi+xi2,train,random=~1|id,
                method="ML")
    fit.REEM<-REEMtree(yi~xi+xi2,train,random=~1|id)
    fit.tree=tree::tree(yi~xi+xi2,train)
    fit.bag =randomForest(yi~xi+xi2,train,mtry=2, importance =TRUE)
    fit.rf =randomForest(yi~xi+xi2,train,mtry=1, importance =TRUE)
    fit.boost =gbm(yi~xi+xi2,train, distribution=
                     "gaussian",n.trees =2000,interaction.depth =4)
    fit.svm=svm(yi~xi+xi2,train, kernel="linear")
    fit.svmkernel=svm(yi~xi+xi2,train, kernel="polynomial",degree=2)
    nndata=model.matrix(~yi+xi+time,train)
    nndata=as.data.frame(nndata)
    fit.nn=neuralnet(yi~xi+xi2,train, hidden=1,linear.output = TRUE,stepmax = 1e6,threshold = 0.1)
      
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

    ni=length(test$yi)
    RMSE.lm[s]=sqrt(sum((test$yi-a.lm)^2)/ni)
    RMSE.lme[s]=sqrt(sum((test$yi-a.lme)^2)/ni)
    RMSE.REEM[s]=sqrt(sum((test$yi-a.REEM)^2)/ni)
    RMSE.tree[s]=sqrt(sum((test$yi-a.tree)^2)/ni)
    RMSE.bag[s]=sqrt(sum((test$yi-a.bag)^2)/ni)
    RMSE.rf[s]=sqrt(sum((test$yi-a.rf)^2)/ni)
    RMSE.boost[s]=sqrt(sum((test$yi-a.boost)^2)/ni)
    RMSE.svm[s]=sqrt(sum((test$yi-a.svm)^2)/ni)
    RMSE.svmkernel[s]=sqrt(sum((test$yi-a.svmkernel)^2)/ni)
    RMSE.nn[s]=sqrt(sum((test$yi-a.nn$net.result)^2)/ni)
    
    BIAS.lm[s]=sum(a.lm-test$yi)/ni
    BIAS.lme[s]=sum(a.lme-test$yi)/ni
    BIAS.REEM[s]=sum(a.REEM-test$yi)/ni
    BIAS.tree[s]=sum(a.tree-test$yi)/ni
    BIAS.bag[s]=sum(a.bag-test$yi)/ni
    BIAS.rf[s]=sum(a.rf-test$yi)/ni
    BIAS.boost[s]=sum(a.boost-test$yi)/ni
    BIAS.svm[s]=sum(a.svm-test$yi)/ni
    BIAS.svmkernel[s]=sum(a.svmkernel-test$yi)/ni
    BIAS.nn[s]=sum(a.nn$net.result-test$yi)/ni
    
    RMSE1.lm[s]=sqrt(sum((test$mu-a.lm)^2)/ni)
    RMSE1.lme[s]=sqrt(sum((test$mu-a.lme)^2)/ni)
    RMSE1.REEM[s]=sqrt(sum((test$mu-a.REEM)^2)/ni)
    RMSE1.tree[s]=sqrt(sum((test$mu-a.tree)^2)/ni)
    RMSE1.bag[s]=sqrt(sum((test$mu-a.bag)^2)/ni)
    RMSE1.rf[s]=sqrt(sum((test$mu-a.rf)^2)/ni)
    RMSE1.boost[s]=sqrt(sum((test$mu-a.boost)^2)/ni)
    RMSE1.svm[s]=sqrt(sum((test$mu-a.svm)^2)/ni)
    RMSE1.svmkernel[s]=sqrt(sum((test$mu-a.svmkernel)^2)/ni)
    RMSE1.nn[s]=sqrt(sum((test$mu-a.nn$net.result)^2)/ni)
}


save.image("data_exc_fo_W.RData")







