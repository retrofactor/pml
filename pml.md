Exercise Manner Prediction based on HAR Dataset
========================================================

This analysis builds a model for predicting Exercise Manner using HAR data from http://groupware.les.inf.puc-rio.br/har for the purpose of Coursera's Practical Machine Learning class.

Initialization
--------------
The training of the model is performed using `caret` package. 
In order to speed-up training phase `doMC` package is also loaded.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(cores = 4)
```

The seed is set to an arbitrary value.

```r
set.seed(12345)
```

Data preprocessing
------------------
The data is obtained with the following commands.


```r
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', destfile='pml-training.csv', method="curl")
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', destfile='pml-testing.csv', method="curl")
```

```r
pml.training <- read.csv('pml-training.csv', header=T)
pml.testing <- read.csv('pml-testing.csv', header=T)
```

A clean dataset is prepared by filtering out variables that can be derived from the other ones (e.g. `kurtosis`, `skewness`) and the `X` variable that enumerates samples.


```r
skipped.cols <- '^stddev_|^var_|^kurtosis_|^skewness_|^amplitude_|^min_|^max_|^avg_|^X$|^cvtd_'

cleaned.training <- pml.training[, -grep(skipped.cols, names(pml.training))]
cleaned.testing<- pml.testing[, -grep(skipped.cols, names(pml.testing))]
```

The `cleaned.training` dataset is split into two sets - `training` and `testing`. 


```r
trainingIndex <- createDataPartition(cleaned.training$classe, p=0.75, list=F)
training <- cleaned.training[trainingIndex,]
testing <- cleaned.training[-trainingIndex,]
```

The `training` set will be used to train the model.  The `testing` set will be used to estimate out-of-sample error.

Training and Parameter Estimation
---------------------------------
For the purpose of prediction the random forest model has been chosen since it is a state-of-art statistical learning method.

First, we perform *5*-fold cross-validation training to estimate the `mtry` parameter of the random forest model.

```r
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5, verboseIter = T)
dev.model <- train(classe ~ ., method="rf", data=training, trControl=fitControl)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 31 on full training set
```

Having the model trained we estimate out-of-sample error by predicting the values using the held-out data from our `testing` set and measuring accuracy. 


```r
testing.predicted <- predict(dev.model, testing)
```
As can be read from the confusion matrix below, accuracy is satisfactory on the testing set.


```r
cm <- confusionMatrix(testing.predicted, testing$classe)
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  947    2    0    0
##          C    0    1  853    0    0
##          D    0    0    0  804    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                     
##                Accuracy : 0.999     
##                  95% CI : (0.998, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 0.999     
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.998    0.998    1.000    1.000
## Specificity             1.000    0.999    1.000    1.000    1.000
## Pos Pred Value          0.999    0.998    0.999    1.000    1.000
## Neg Pred Value          1.000    0.999    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.285    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.999    0.999    1.000    1.000
```

The out-of-sample error rate is as follows.

```r
1 - cm$overall[["Accuracy"]]
```

```
## [1] 0.0008157
```

Prediction
-----------
For the purpose of predicting labels for the test cases from the `pml.testing` dataset the final model is build using all the available data by performing the same cross-validation procedure as above.


```r
final.model <- train(classe ~ ., method="rf", data=cleaned.training, trControl=fitControl)
```

```
## + Fold1.Rep1: mtry= 2 
## - Fold1.Rep1: mtry= 2 
## + Fold1.Rep1: mtry=31 
## - Fold1.Rep1: mtry=31 
## + Fold1.Rep1: mtry=61 
## - Fold1.Rep1: mtry=61 
## + Fold2.Rep1: mtry= 2 
## - Fold2.Rep1: mtry= 2 
## + Fold2.Rep1: mtry=31 
## - Fold2.Rep1: mtry=31 
## + Fold2.Rep1: mtry=61 
## - Fold2.Rep1: mtry=61 
## + Fold3.Rep1: mtry= 2 
## - Fold3.Rep1: mtry= 2 
## + Fold3.Rep1: mtry=31 
## - Fold3.Rep1: mtry=31 
## + Fold3.Rep1: mtry=61 
## - Fold3.Rep1: mtry=61 
## + Fold4.Rep1: mtry= 2 
## - Fold4.Rep1: mtry= 2 
## + Fold4.Rep1: mtry=31 
## - Fold4.Rep1: mtry=31 
## + Fold4.Rep1: mtry=61 
## - Fold4.Rep1: mtry=61 
## + Fold5.Rep1: mtry= 2 
## - Fold5.Rep1: mtry= 2 
## + Fold5.Rep1: mtry=31 
## - Fold5.Rep1: mtry=31 
## + Fold5.Rep1: mtry=61 
## - Fold5.Rep1: mtry=61 
## + Fold1.Rep2: mtry= 2 
## - Fold1.Rep2: mtry= 2 
## + Fold1.Rep2: mtry=31 
## - Fold1.Rep2: mtry=31 
## + Fold1.Rep2: mtry=61 
## - Fold1.Rep2: mtry=61 
## + Fold2.Rep2: mtry= 2 
## - Fold2.Rep2: mtry= 2 
## + Fold2.Rep2: mtry=31 
## - Fold2.Rep2: mtry=31 
## + Fold2.Rep2: mtry=61 
## - Fold2.Rep2: mtry=61 
## + Fold3.Rep2: mtry= 2 
## - Fold3.Rep2: mtry= 2 
## + Fold3.Rep2: mtry=31 
## - Fold3.Rep2: mtry=31 
## + Fold3.Rep2: mtry=61 
## - Fold3.Rep2: mtry=61 
## + Fold4.Rep2: mtry= 2 
## - Fold4.Rep2: mtry= 2 
## + Fold4.Rep2: mtry=31 
## - Fold4.Rep2: mtry=31 
## + Fold4.Rep2: mtry=61 
## - Fold4.Rep2: mtry=61 
## + Fold5.Rep2: mtry= 2 
## - Fold5.Rep2: mtry= 2 
## + Fold5.Rep2: mtry=31 
## - Fold5.Rep2: mtry=31 
## + Fold5.Rep2: mtry=61 
## - Fold5.Rep2: mtry=61 
## + Fold1.Rep3: mtry= 2 
## - Fold1.Rep3: mtry= 2 
## + Fold1.Rep3: mtry=31 
## - Fold1.Rep3: mtry=31 
## + Fold1.Rep3: mtry=61 
## - Fold1.Rep3: mtry=61 
## + Fold2.Rep3: mtry= 2 
## - Fold2.Rep3: mtry= 2 
## + Fold2.Rep3: mtry=31 
## - Fold2.Rep3: mtry=31 
## + Fold2.Rep3: mtry=61 
## - Fold2.Rep3: mtry=61 
## + Fold3.Rep3: mtry= 2 
## - Fold3.Rep3: mtry= 2 
## + Fold3.Rep3: mtry=31 
## - Fold3.Rep3: mtry=31 
## + Fold3.Rep3: mtry=61 
## - Fold3.Rep3: mtry=61 
## + Fold4.Rep3: mtry= 2 
## - Fold4.Rep3: mtry= 2 
## + Fold4.Rep3: mtry=31 
## - Fold4.Rep3: mtry=31 
## + Fold4.Rep3: mtry=61 
## - Fold4.Rep3: mtry=61 
## + Fold5.Rep3: mtry= 2 
## - Fold5.Rep3: mtry= 2 
## + Fold5.Rep3: mtry=31 
## - Fold5.Rep3: mtry=31 
## + Fold5.Rep3: mtry=61 
## - Fold5.Rep3: mtry=61 
## + Fold1.Rep4: mtry= 2 
## - Fold1.Rep4: mtry= 2 
## + Fold1.Rep4: mtry=31 
## - Fold1.Rep4: mtry=31 
## + Fold1.Rep4: mtry=61 
## - Fold1.Rep4: mtry=61 
## + Fold2.Rep4: mtry= 2 
## - Fold2.Rep4: mtry= 2 
## + Fold2.Rep4: mtry=31 
## - Fold2.Rep4: mtry=31 
## + Fold2.Rep4: mtry=61 
## - Fold2.Rep4: mtry=61 
## + Fold3.Rep4: mtry= 2 
## - Fold3.Rep4: mtry= 2 
## + Fold3.Rep4: mtry=31 
## - Fold3.Rep4: mtry=31 
## + Fold3.Rep4: mtry=61 
## - Fold3.Rep4: mtry=61 
## + Fold4.Rep4: mtry= 2 
## - Fold4.Rep4: mtry= 2 
## + Fold4.Rep4: mtry=31 
## - Fold4.Rep4: mtry=31 
## + Fold4.Rep4: mtry=61 
## - Fold4.Rep4: mtry=61 
## + Fold5.Rep4: mtry= 2 
## - Fold5.Rep4: mtry= 2 
## + Fold5.Rep4: mtry=31 
## - Fold5.Rep4: mtry=31 
## + Fold5.Rep4: mtry=61 
## - Fold5.Rep4: mtry=61 
## + Fold1.Rep5: mtry= 2 
## - Fold1.Rep5: mtry= 2 
## + Fold1.Rep5: mtry=31 
## - Fold1.Rep5: mtry=31 
## + Fold1.Rep5: mtry=61 
## - Fold1.Rep5: mtry=61 
## + Fold2.Rep5: mtry= 2 
## - Fold2.Rep5: mtry= 2 
## + Fold2.Rep5: mtry=31 
## - Fold2.Rep5: mtry=31 
## + Fold2.Rep5: mtry=61 
## - Fold2.Rep5: mtry=61 
## + Fold3.Rep5: mtry= 2 
## - Fold3.Rep5: mtry= 2 
## + Fold3.Rep5: mtry=31 
## - Fold3.Rep5: mtry=31 
## + Fold3.Rep5: mtry=61 
## - Fold3.Rep5: mtry=61 
## + Fold4.Rep5: mtry= 2 
## - Fold4.Rep5: mtry= 2 
## + Fold4.Rep5: mtry=31 
## - Fold4.Rep5: mtry=31 
## + Fold4.Rep5: mtry=61 
## - Fold4.Rep5: mtry=61 
## + Fold5.Rep5: mtry= 2 
## - Fold5.Rep5: mtry= 2 
## + Fold5.Rep5: mtry=31 
## - Fold5.Rep5: mtry=31 
## + Fold5.Rep5: mtry=61 
## - Fold5.Rep5: mtry=61 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 31 on full training set
```

The model predicts the following classes for the `pml.testing` dataset.


```r
pml.testing.predicted <- predict(final.model, cleaned.testing)
pml.testing.predicted
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Finally, the submission files are generated using the procedure `pml_write_files` taken from the PML course site.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pml.testing.predicted)
```
