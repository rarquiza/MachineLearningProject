# Final
library(caret)
activity <- read.csv(file="pml-training.csv", header=TRUE, sep=",")
str(activity)

# Check actual classifications
table(activity$classe)

# Pre-process
# Exclude the index, user name, timestamps, and features with mostly NA values
activity <- activity[,c(8:11,37:49,60:68,84:86,102,113:124,140,151:159,160)]
set.seed(1234)
inBuild <- createDataPartition(y = activity$classe, p=0.7, list=FALSE)
validation <- activity[-inBuild,]
buildData <- activity[inBuild,]
inTrain <- createDataPartition(y = buildData$classe, p=0.7, list=FALSE)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]

# Feature plot to see correlation and patterns for a few features
png("FeaturePlot.png", width=560, height=560)
featurePlot(x=training[,1:4],y=training$classe,plot="pairs")
dev.off()

# Classification and Regression Trees (CART)
library(rpart); library(rpart.plot)
modCART <- rpart(classe ~ ., data=training)

png("CART.png", width=700, height=560)
prp(modCART)
dev.off()

# Confusion Matrix and Accuracy
table(predict(modCART,training,type="class"),training$classe)
(2258+1328+1267+951+1144)/9619
[1] 0.7223204

# Predictions to the test set
table(predict(modCART,newdata=testing,type="class"),testing$classe)
(935+569+529+394+487)/4118
[1] 0.7076251

# Random Forest
set.seed(1234)
modRF <- train(classe ~., data=training, method="rf", trControl = trainControl(method="cv",number=5,repeats=5))

# Confusion Matrix and Accuracy
table(predict(modRF$finalModel,training),training$classe)
(2735+1861+1678+1577+1768)/9619
[1] 1

# Predictions to the test set
table(predict(modRF$finalModel,newdata=testing),testing$classe)
(1170+787+712+666+749)/4118
[1] 0.9917436


# Predictions to the validation set
table(predict(modRF$finalModel,newdata=validation),validation$classe)
(1669+1121+1013+951+1075)/5885
[1] 0.9904843


