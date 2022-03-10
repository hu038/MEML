rm(list=ls())
.libPaths(c("C:/Rlib"))
###read dataset
library(nlme)
data <- Milk
dim(data)
head(data)
## check if there are missing values in this dataset
data <- data[!is.na(data$protein),] 
table(data$Diet)

####piecewise linear model, choose the breakpoint
bs=seq(1,6,0.1)
mse <- rep(0,length(bs))
for(j in 1:length(bs)){
  i=bs[j]
  dummyknot=rep(0,length(data$Time))
  ###TDN is the (t-3)*D
  dummyknot[data$Time>i]=1
  data$tdif=data$Time-i
  data$DN=dummyknot
  data$TDN=dummyknot*data$tdif
  ###TDN2 is the (t-3)^2*D
  data$tdif2=(data$tdif)^2
  data$TDN2=data$DN*data$tdif2
  ###onet is the t
  data$DNN=i*data$DN/data$Time
  data$DNN[data$Time<=i]=1
  data$onet=data$Time*data$DNN
  
  fit=lm(protein~Diet+onet+TDN,data=data)  ##Diggle without correlation 6 parameters
  #fit=gls(protein~Diet+onet+TDN,data=data,
  #method="ML",control = list(singular.ok = TRUE))
  mse[j] <- deviance(fit)
  #mse[j] <- summary(fit)$sigma
  #print(mse)
}
mse <- as.numeric(mse)
bs[which(mse==min(mse))]

#The results show that according to the values of mean square error (sum of residuals),
#we can choose 2.6 as the breakpoint.
dummyknot=rep(0,length(data$Time))
dummyknot[data$Time>2.6]=1
data$tdif=data$Time-2.6
data$DN=dummyknot
data$TDN=dummyknot*data$tdif
data$tdif2=(data$tdif)^2
data$TDN2=data$DN*data$tdif2
data$DNN=2.6*data$DN/data$Time
data$DNN[data$Time<=2.6]=1
data$onet=data$Time*data$DNN


library(nlme)
library(tree)
library(randomForest)
library (gbm)
library(e1071)
library(neuralnet)
library(REEMtree)
library(caret)

###one-step prediction
RMSE.svmk=RMSE.my=RMSE.lm=RMSE.mylm=RMSE.lmm=RMSE.gls=RMSE.glsp=RMSE.mygls=RMSE.REEM1=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=rep(0,12)
NMSE.svmk=NMSE.my=NMSE.lm=NMSE.mylm=NMSE.lmm=NMSE.gls=NMSE.glsp=NMSE.mygls=NMSE.REEM1=NMSE.REEM=NMSE.tree=NMSE.bag=NMSE.rf=NMSE.boost=NMSE.svm=NMSE.nn=rep(0,12)

for(j in 8:19){
  subset=(data$Time<j)
  testset=(data$Time==j)
  train <- data[subset,] 
  test <- data[testset,]
  fit.my=lm(protein~Diet+onet+TDN,data=train) 
  fit.lm=lme(protein~Diet+onet+TDN,data=train,random=~1|Cow,
             method="ML")
  fit.mylm=lme(protein~Diet+onet+TDN,data=train,random=~Time+Diet|Cow,
               method="ML")
  fit.lmm=lme(protein~Time+Diet,data=train,random=~Time+Diet|Cow,
              method="ML")
  fit.glsp=gls(protein~Diet+onet+TDN,data=train,
               correlation=corCompSymm(form=~1|Cow),
               method="ML")
  fit.mygls=gls(protein~Diet+onet+TDN,data=train,
                correlation=corCAR1(form=~Time|Cow),
                method="ML")
  fit.gls=gls(protein~Diet+onet+TDN+TDN2,data=train,
              correlation=corCAR1(form=~Time|Cow),
              method="ML")
  fit.REEM1<-REEMtree(protein~Diet+onet+TDN,data=train,random=~1|Cow)
  fit.REEM<-REEMtree(protein~Diet+onet+TDN,data=train,random=~Time+Diet|Cow)
  fit.tree=tree::tree(protein~Time+Diet,train)
  fit.bag =randomForest(protein~Time+Diet,train,mtry=2, importance =TRUE)
  fit.rf =randomForest(protein~Time+Diet,train,mtry=1, importance =TRUE)
  fit.boost =gbm(protein~Time+Diet,train, distribution=
                   "gaussian",n.trees =5000,interaction.depth =4)
  fit.svm=svm(protein~Time+Diet,train, kernel="linear")
  fit.svmk=svm(protein~Time+Diet,train, kernel="polynomial")
  nndata=model.matrix(~protein+Time+Diet,train)
  colnames(nndata)[4]="Dietbl"
  fit.nn=neuralnet(protein~Time+Dietbl+Dietlupins,nndata,hidden=3,linear.output = TRUE,stepmax=1e6)
  #fit.nn=train(protein~Time+Dietbl+Dietlupins,nndata,method='nnet',linout=TRUE,trace=FALSE)
  
  a.my=predict(fit.my,test)
  a.lm=predict(fit.lm,test)
  a.mylm=predict(fit.mylm,test)
  a.lmm=predict(fit.lmm, test)
  a.gls=predict(fit.gls, test)
  a.glsp=predict(fit.glsp, test)
  a.mygls=predict(fit.mygls, test)
  a.REEM1=predict(fit.REEM1, test,id=test$Cow, EstimateRandomEffects=TRUE)
  a.REEM=predict(fit.REEM, test,id=test$Cow, EstimateRandomEffects=TRUE)
  a.tree=predict(fit.tree, test)
  a.bag=predict(fit.bag, test)
  a.rf=predict(fit.rf, test)
  a.boost=predict(fit.boost, test,n.trees =5000)
  a.svm=predict(fit.svm, test)
  a.svmk=predict(fit.svmk, test)
  nndatatest=model.matrix(~protein+Time+Diet,test)
  colnames(nndatatest)[4]="Dietbl"
  a.nn=compute(fit.nn,nndatatest)
  
  ni=length(test$protein)
  
  RMSE.my[j-7]=sqrt(sum((test$protein-a.my)^2)/ni)
  RMSE.lm[j-7]=sqrt(sum((test$protein-a.lm)^2)/ni)
  RMSE.mylm[j-7]=sqrt(sum((test$protein-a.mylm)^2)/ni)
  RMSE.lmm[j-7]=sqrt(sum((test$protein-a.lmm)^2)/ni)
  RMSE.gls[j-7]=sqrt(sum((test$protein-a.gls)^2)/ni)
  RMSE.glsp[j-7]=sqrt(sum((test$protein-a.glsp)^2)/ni)
  RMSE.mygls[j-7]=sqrt(sum((test$protein-a.mygls)^2)/ni)
  RMSE.REEM[j-7]=sqrt(sum((test$protein-a.REEM)^2)/ni)
  RMSE.REEM1[j-7]=sqrt(sum((test$protein-a.REEM1)^2)/ni)
  RMSE.tree[j-7]=sqrt(sum((test$protein-a.tree)^2)/ni)
  RMSE.bag[j-7]=sqrt(sum((test$protein-a.bag)^2)/ni)
  RMSE.rf[j-7]=sqrt(sum((test$protein-a.rf)^2)/ni)
  RMSE.boost[j-7]=sqrt(sum((test$protein-a.boost)^2)/ni)
  RMSE.svm[j-7]=sqrt(sum((test$protein-a.svm)^2)/ni)
  RMSE.svmk[j-7]=sqrt(sum((test$protein-a.svmk)^2)/ni)
  RMSE.nn[j-7]=sqrt(sum((nndatatest[,2]-a.nn$net.result)^2)/ni)
  
  MSE=sum((test$protein-mean(test$protein))^2)
  
  NMSE.my[j-7]=sum((test$protein-a.my)^2)/MSE
  NMSE.lm[j-7]=sum((test$protein-a.lm)^2)/MSE
  NMSE.mylm[j-7]=sum((test$protein-a.mylm)^2)/MSE
  NMSE.lmm[j-7]=sum((test$protein-a.lmm)^2)/MSE
  NMSE.gls[j-7]=sum((test$protein-a.gls)^2)/MSE
  NMSE.glsp[j-7]=sum((test$protein-a.glsp)^2)/MSE
  NMSE.mygls[j-7]=sum((test$protein-a.mygls)^2)/MSE
  NMSE.REEM1[j-7]=sum((test$protein-a.REEM1)^2)/MSE
  NMSE.REEM[j-7]=sum((test$protein-a.REEM)^2)/MSE
  NMSE.tree[j-7]=sum((test$protein-a.tree)^2)/MSE
  NMSE.bag[j-7]=sum((test$protein-a.bag)^2)/MSE
  NMSE.rf[j-7]=sum((test$protein-a.rf)^2)/MSE
  NMSE.boost[j-7]=sum((test$protein-a.boost)^2)/MSE
  NMSE.svm[j-7]=sum((test$protein-a.svm)^2)/MSE
  NMSE.svmk[j-7]=sum((test$protein-a.svmk)^2)/MSE
  NMSE.nn[j-7]=sum((nndatatest[,2]-a.nn$net.result)^2)/MSE
}

library(knitr)

NMSE_onestep=cbind(NMSE.my,NMSE.lm,NMSE.mylm,NMSE.lmm,NMSE.gls,NMSE.glsp,NMSE.mygls,NMSE.REEM1,NMSE.REEM,
             NMSE.tree, NMSE.bag,NMSE.rf,NMSE.boost,NMSE.svm,NMSE.svmk,NMSE.nn)
NMSE_onestep=data.frame(NMSE_onestep)

save(NMSE_onestep,file="NMSE_onestep_milk.RData")

RMSE_onestep=cbind(RMSE.my,RMSE.lm,RMSE.mylm,RMSE.lmm,RMSE.gls,RMSE.glsp,RMSE.mygls,RMSE.REEM1,RMSE.REEM,
                   RMSE.tree, RMSE.bag,RMSE.rf,RMSE.boost,RMSE.svm,RMSE.svmk,RMSE.nn)
RMSE_onestep=data.frame(RMSE_onestep)

save(RMSE_onestep,file="RMSE_onestep_milk.RData")

mean_NMSE_onestepm=apply(NMSE_onestep,2,mean)
mean_RMSE_onestepm=apply(RMSE_onestep,2,mean)

##what I need the models for comparision
RMSE_onestep_s=RMSE_onestep[,c(3,5,8,10,11,12,13,14,15,16)]

knitr::kable(round(RMSE_onestep_s,3),format="latex")

mean_RMSE_onestep_s=apply(RMSE_onestep_s,2,mean)

M=c("plme","diggle","re-em tree","tree","bag","rf","boost","svm","svmk","nn")
barplot(mean_RMSE_onestep_s,names.arg = M,xlab="Method",ylab="RMSE value",col="blue",
        border="red")

# ############### Plot ########################
# max(result_onestep)
# min(result_onestep)
# 
# x11()
# plot(NMSE.mylm,xlab="The prediction week",xaxt="n",ylab="NMSE",
#      ylim=c(min(result_onestep),max(result_onestep)),
#      main="The NMSE value in one step prediction",pch=1)
# axis(1,at=c(1:12),labels=c("week8","week9","week10","week11","week12","week13","week14","week15","week16",
#          "week17","week18","week19"))
# lines(NMSE.mylm,col=1)
# points(NMSE.gls,pch=2)
# lines(NMSE.gls,col=2)
# 
# points(NMSE.tree,pch=3)
# lines(NMSE.tree,col=3)
# points(NMSE.bag,pch=4)
# lines(NMSE.bag,col=4)
# points(NMSE.rf,pch=5)
# lines(NMSE.rf,col=5)
# points(NMSE.boost,pch=6)
# lines(NMSE.boost,col=6)
# points(NMSE.svm,pch=7)
# lines(NMSE.svm,col=7)
# points(NMSE.nn,pch=8)
# lines(NMSE.nn,col=8)
# points(NMSE.REEM1,pch=9)
# lines(NMSE.REEM1,col=9)
# points(NMSE.REEM,pch=10)
# lines(NMSE.REEM,col=10)
# 
# legend("topright",cex=0.7,lwd=2,c("mylm(=1)","gls(=2)","tree(=3)","bag(=4)",
#                           "rf(=5)","boost(=6)","svm(=7)","nn(=8)","re-em0(=9)","re-em(=10)"),
#        col=c(1:10),pch=c(1:10))


#########two-step presiction
RMSE.my=RMSE.lm=RMSE.mylm=RMSE.lmm=RMSE.gls=RMSE.glsp=RMSE.mygls=RMSE.REEM1=RMSE.REEM=RMSE.tree=RMSE.bag=RMSE.rf=RMSE.boost=RMSE.svm=RMSE.nn=rep(0,11)
NMSE.my=NMSE.lm=NMSE.mylm=NMSE.lmm=NMSE.gls=NMSE.glsp=NMSE.mygls=NMSE.REEM1=NMSE.REEM=NMSE.tree=NMSE.bag=NMSE.rf=NMSE.boost=NMSE.svm=NMSE.nn=rep(0,11)

for(j in 8:18){
  subset=(data$Time<j)
  testset=(data$Time==j+1)
  train <- data[subset,] 
  test <- data[testset,]
  fit.my=lm(protein~Diet+onet+TDN,data=train) 
  fit.lm=lme(protein~Diet+onet+TDN,data=train,random=~1|Cow,
             method="ML")
  fit.mylm=lme(protein~Diet+onet+TDN,data=train,random=~Time+Diet|Cow,
               method="ML")
  fit.lmm=lme(protein~Time+Diet,data=train,random=~Time+Diet|Cow,
              method="ML")
  fit.glsp=gls(protein~Diet+onet+TDN,data=train,
               correlation=corCompSymm(form=~1|Cow),
               method="ML")
  fit.mygls=gls(protein~Diet+onet+TDN,data=train,
                correlation=corCAR1(form=~Time|Cow),
                method="ML")
  fit.gls=gls(protein~Diet+onet+TDN+TDN2,data=train,
              correlation=corCAR1(form=~Time|Cow),
              method="ML")
  fit.REEM1<-REEMtree(protein~Diet+onet+TDN,data=train,random=~1|Cow)
  fit.REEM<-REEMtree(protein~Diet+onet+TDN,data=train,random=~Time+Diet|Cow)
  fit.tree=tree::tree(protein~Time+Diet,train)
  fit.bag =randomForest(protein~Time+Diet,train,mtry=2, importance =TRUE)
  fit.rf =randomForest(protein~Time+Diet,train,mtry=1, importance =TRUE)
  fit.boost =gbm(protein~Time+Diet,train, distribution=
                   "gaussian",n.trees =5000,interaction.depth =4)
  fit.svm=svm(protein~Time+Diet,train, kernel="linear", scale=FALSE)
  nndata=model.matrix(~protein+Time+Diet,train)
  colnames(nndata)[4]="Dietbl"
  fit.nn=neuralnet(protein~Time+Dietbl+Dietlupins,nndata,hidden=3,linear.output = TRUE)
  
  a.my=predict(fit.my,test)
  a.lm=predict(fit.lm,test)
  a.mylm=predict(fit.mylm,test)
  a.lmm=predict(fit.lmm, test)
  a.gls=predict(fit.gls, test)
  a.glsp=predict(fit.glsp, test)
  a.mygls=predict(fit.mygls, test)
  a.REEM1=predict(fit.REEM1, test,id=test$Cow, EstimateRandomEffects=TRUE)
  a.REEM=predict(fit.REEM, test,id=test$Cow, EstimateRandomEffects=TRUE)
  a.tree=predict(fit.tree, test)
  a.bag=predict(fit.bag, test)
  a.rf=predict(fit.rf, test)
  a.boost=predict(fit.boost, test,n.trees =5000)
  a.svm=predict(fit.svm, test)
  nndatatest=model.matrix(~protein+Time+Diet,test)
  colnames(nndatatest)[4]="Dietbl"
  a.nn=compute(fit.nn,nndatatest)
  
  ni=length(test$protein)
  
  RMSE.my[j-7]=sqrt(sum((test$protein-a.my)^2)/ni)
  RMSE.lm[j-7]=sqrt(sum((test$protein-a.lm)^2)/ni)
  RMSE.mylm[j-7]=sqrt(sum((test$protein-a.mylm)^2)/ni)
  RMSE.lmm[j-7]=sqrt(sum((test$protein-a.lmm)^2)/ni)
  RMSE.gls[j-7]=sqrt(sum((test$protein-a.gls)^2)/ni)
  RMSE.glsp[j-7]=sqrt(sum((test$protein-a.glsp)^2)/ni)
  RMSE.mygls[j-7]=sqrt(sum((test$protein-a.mygls)^2)/ni)
  RMSE.REEM[j-7]=sqrt(sum((test$protein-a.REEM)^2)/ni)
  RMSE.REEM1[j-7]=sqrt(sum((test$protein-a.REEM1)^2)/ni)
  RMSE.tree[j-7]=sqrt(sum((test$protein-a.tree)^2)/ni)
  RMSE.bag[j-7]=sqrt(sum((test$protein-a.bag)^2)/ni)
  RMSE.rf[j-7]=sqrt(sum((test$protein-a.rf)^2)/ni)
  RMSE.boost[j-7]=sqrt(sum((test$protein-a.boost)^2)/ni)
  RMSE.svm[j-7]=sqrt(sum((test$protein-a.svm)^2)/ni)
  RMSE.nn[j-7]=sqrt(sum((nndatatest[,2]-a.nn$net.result)^2)/ni)
  
  
  MSE=sum((test$protein-mean(test$protein))^2)
  
  NMSE.my[j-7]=sum((test$protein-a.my)^2)/MSE
  NMSE.lm[j-7]=sum((test$protein-a.lm)^2)/MSE
  NMSE.mylm[j-7]=sum((test$protein-a.mylm)^2)/MSE
  NMSE.lmm[j-7]=sum((test$protein-a.lmm)^2)/MSE
  NMSE.gls[j-7]=sum((test$protein-a.gls)^2)/MSE
  NMSE.glsp[j-7]=sum((test$protein-a.glsp)^2)/MSE
  NMSE.mygls[j-7]=sum((test$protein-a.mygls)^2)/MSE
  NMSE.REEM1[j-7]=sum((test$protein-a.REEM1)^2)/MSE
  NMSE.REEM[j-7]=sum((test$protein-a.REEM)^2)/MSE
  NMSE.tree[j-7]=sum((test$protein-a.tree)^2)/MSE
  NMSE.bag[j-7]=sum((test$protein-a.bag)^2)/MSE
  NMSE.rf[j-7]=sum((test$protein-a.rf)^2)/MSE
  NMSE.boost[j-7]=sum((test$protein-a.boost)^2)/MSE
  NMSE.svm[j-7]=sum((test$protein-a.svm)^2)/MSE
  NMSE.nn[j-7]=sum((nndatatest[,2]-a.nn$net.result)^2)/MSE
}

library(knitr)

NMSE_twostep=cbind(NMSE.my,NMSE.lm,NMSE.mylm,NMSE.lmm,NMSE.gls,NMSE.glsp,NMSE.mygls,NMSE.REEM1,NMSE.REEM,
                      NMSE.tree, NMSE.bag,NMSE.rf,NMSE.boost,NMSE.svm,NMSE.nn)
NMSE_twostep=data.frame(NMSE_twostep)
save(NMSE_twostep,file="NMSE_twostep_milk.RData")

RMSE_twostep=cbind(RMSE.my,RMSE.lm,RMSE.mylm,RMSE.lmm,RMSE.gls,RMSE.glsp,RMSE.mygls,RMSE.REEM1,RMSE.REEM,
                   RMSE.tree, RMSE.bag,RMSE.rf,RMSE.boost,RMSE.svm,RMSE.nn)
RMSE_twostep=data.frame(RMSE_twostep)
save(RMSE_twostep,file="RMSE_twostep_milk.RData")

mean_NMSE_twostep=apply(NMSE_twostep,2,mean)
mean_RMSE_twostep=apply(RMSE_twostep,2,mean)

##what I need the models for comparision
RMSE_twostep_s=RMSE_twostep[,c(3,5,8,10,11,12,13,14,15)]

knitr::kable(round(RMSE_twostep_s,3),format="latex")

mean_RMSE_twostep_s=apply(RMSE_twostep_s,2,mean)




M=c("plme","diggle","re-em tree","tree","bag","rf","boost","svm","nn")
barplot(mean_RMSE_twostep_s,names.arg = M,xlab="Method",ylab="RMSE value",col="blue",
        border="red")

# max(result_twostep)
# min(result_twostep)
# 
# plot(NMSE.mylm,xlab="The prediction week",xaxt="n",ylab="NMSE",
#      ylim=c(min(result_twostep),max(result_twostep)),
#      main="The NMSE value in two step prediction",pch=1)
# axis(1,at=c(1:11),labels=c("week9","week10","week11","week12","week13","week14","week15","week16",
#                            "week17","week18","week19"))
# lines(NMSE.mylm,col=1)
# points(NMSE.gls,pch=2)
# lines(NMSE.gls,col=2)
# 
# points(NMSE.tree,pch=3)
# lines(NMSE.tree,col=3)
# points(NMSE.bag,pch=4)
# lines(NMSE.bag,col=4)
# points(NMSE.rf,pch=5)
# lines(NMSE.rf,col=5)
# points(NMSE.boost,pch=6)
# lines(NMSE.boost,col=6)
# points(NMSE.svm,pch=7)
# lines(NMSE.svm,col=7)
# points(NMSE.nn,pch=8)
# lines(NMSE.nn,col=8)
# points(NMSE.REEM0,pch=9)
# lines(NMSE.REEM0,col=9)
# points(NMSE.REEM,pch=10)
# lines(NMSE.REEM,col=10)
# 
# legend("topright",cex=0.7,lwd=2,c("mylm(=1)","gls(=2)","tree(=3)","bag(=4)",
#                                   "rf(=5)","boost(=6)","svm(=7)","nn(=8)","re-em0(=9)","re-em(=10)"),
#        col=c(1:10),pch=c(1:10))


