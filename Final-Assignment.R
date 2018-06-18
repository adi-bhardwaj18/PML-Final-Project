require(caret)
train.dat <- read.csv("E:/ ... /pml-training.csv")
test.dat <- read.csv("E:/ ... /pml-testing.csv")
str(train.dat)

#### ---- Converting Skewness & Kurtosis Variables to Numeric ---- ####
tr.dat <- apply(train.dat[ , 12:159], 2, as.numeric)
tr.dat <- data.frame(tr.dat)
trn.dat <- cbind(train.dat[1:11], tr.dat, train.dat[160])
str(trn.dat)
tst.dat <- apply(test.dat[ , 12:159], 2, as.numeric)
tst.dat <- data.frame(tst.dat)
ts.dat <- cbind(test.dat[1:11], tst.dat, test.dat[160])
str(trn.dat)
rm(train.dat, test.dat, tr.dat, tst.dat)

#### ---- Exploratory Data Analysis ---- ####
summary(trn.dat$X)
# * 'X' is probably index number, a unique identifier for each row. Hence, it is best to exclude this variable from the analysis
summary(trn.dat$user_name)
# * Since the dataset is meant to build a model that is generalizable, username should not be an apt variable, since it can
# * destabilize the model if it is used for any other user name
summary(trn.dat$raw_timestamp_part_1)
# * This is a zero variance variable, and there is no information on what this means. Hence, we will ignore this
summary(trn.dat$raw_timestamp_part_2)
# * No information about what this variable is, but it seems the name means it is related to time. We next plot its relationship
# * with the outcome.
ggplot(trn.dat, aes(x = classe, y = raw_timestamp_part_2)) +
    geom_point()
# * There is virtually no variation among the classes of the response. Hence, we will ignore this variable
summary(trn.dat$new_window)
# * Not possibly an accelerometer variable. Hence, we will ignore this.
summary(trn.dat$num_window)
ggplot(trn.dat, aes(x = classe, y = num_window)) +
    geom_point()
# * The variable doesn't give any information w.r.t the response. Hence, we ignore this variable as well
## ** The variables 'roll', 'pitch' and 'yaw' are mostly related to acceleromters. Hence, we focus on these.
which(colnames(trn.dat) == "num_window")
trn.dat <- trn.dat[-c(1:7)]
ts.dat <- ts.dat[-c(1:7)]

#### ---- NAs ---- ####
# * Next, we focus on the variables in the data that are majority NAs
na.val <- data.frame(sapply(trn.dat, function (x) {sum(is.na(x))}))
colnames(na.val) <- "Sum.NA"
# * We see that the variables either have almost all the data missing, or none. Imputing at such a large scale would jeopardize the
# * integrity of the model. Hence, we next remove all the variables with large missing values

## -- Identifying predictors with NAs > 10 -- ##
na.pred <- vector(mode = "character", length = length(trn.dat))
s <- 0
n <- length(trn.dat)
for (i in 1 : n) {
    if (sum(is.na(trn.dat[i])) > 10) {
        s <- s + 1
        na.pred[s] <- colnames(trn.dat[i])
    }
}
na.pred

## -- Cross-Verifying by counting the na.val rows with predictors having NAs greater than 10 -- ##
z = 0
for (i in 1 : nrow(na.val)) {
    if (na.val[i, 1] > 0)
        z <- z + 1
}
rm(na.val, i, n, s, z)
# * Since the counters 's' and 'z' both are at 100, we can state that our function captured all the predictors with high number of
# * NAs. We can proceed to delete these from the dataset
na.pred <- na.pred[1:100]       # > We shorten the length of the vector to eliminate the spaces not filled 
n <- length(na.pred)
pred.col.num <- vector(mode = "numeric", length = n)
for (i in 1:n) {
    pred.col.num[i] <- which(colnames(trn.dat) == na.pred[i])    
}
rm(i, n)
trn.dat <- trn.dat[-pred.col.num]
ts.dat <- ts.dat[-pred.col.num]

rm(na.pred, pred.col.num)


## ** Our dataset is now reduced to 53 columns, having 52 predictors
## ** We now split the data into training, validation data and testing datasets
set.seed(2711)
sampler1 <- sample(x = seq(1:19622), size = 0.75 * nrow(trn.dat), replace = FALSE)
tr.dat <- trn.dat[sampler1, ]
val.dat <- trn.dat[-sampler1, ]


## ** We now begin the explanatory data analysis of the remaining variables
# - Five-Number Summary - #
require(lolcat)
# -> 'lolcat' is a custom R-package for certain tasks that I have used in my courses. You can download it using the following
# -> commands:
# -> require(devtools)
# -> install_github ("burrm/lolcat")
sum.out <- summary.impl(fx = tr.dat[-53], stat.n = T, stat.miss = T, stat.min = T, stat.q1 = T, stat.median = T, stat.q3 = T,
                        stat.max = T)
summary(tr.dat$classe)

## -- Prediction using Random Forest to assess important predictors -- ##
require(randomForest)
tr.out <- as.vector(tr.dat$classe)
tr.out <- as.factor(tr.out)
RF.mod <- randomForest(x = tr.dat[-53], y = tr.out, importance = TRUE)
imp.RF <- importance(RF.mod)
imp.DF <- data.frame(Variables = row.names(imp.RF), Mean.Dec.Gini = imp.RF[ ,7])
imp.DF <- imp.DF[order(imp.DF$Mean.Dec.Gini, decreasing = TRUE), ]
ggplot(imp.DF[1:20, ], aes(x = reorder(Variables, Mean.Dec.Gini), y = Mean.Dec.Gini)) +
    geom_bar(stat = "identity") +
    labs(x = "Variables", y = "Mean Decrease in Gini if Variable Randomly Selected") +
    coord_flip() +
    theme(legend.position = "none")

## -- Prediction on the Validation Data -- ##
pred.out <- predict(object = RF.mod, newdata = val.dat)
confusionMatrix(data = pred.out, reference = val.dat$classe)
pred2 <- predict(object = RF.mod, ts.dat)
ts.dat <- cbind(ts.dat, pred2)
