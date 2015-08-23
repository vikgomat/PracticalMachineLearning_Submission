library(caret)
library(randomForest)
library(Hmisc)

library(foreach)
library(doParallel)
set.seed(1234)
data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!"))
clData <- data
for(i in c(8:ncol(clData)-1)) {clData[,i] = as.numeric(as.character(clData[,i]))}
featuresnames <- colnames(clData[colSums(is.na(clData)) == 0])[-(1:7)]
features <- clData[featuresnames]
xdata <- createDataPartition(y=features$classe, p=3/4, list=FALSE )
training <- features[xdata,]
testing <- features[-xdata,]
model <- foreach(ntree=rep(150, 4), .combine=randomForest::combine) %dopar% randomForest(training[-ncol(training)], training$classe, ntree=ntree)
predictionsTr <- predict(model, newdata=training)
confusionMatrix(predictionsTr,training$classe)
predictionsTe <- predict(model, newdata=testing)
confusionMatrix(predictionsTe,testing$classe)