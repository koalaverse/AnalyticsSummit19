# Setup ------------------------------------------------------------------------

# Load required packages
library(caret)         # for data splitting function
library(gbm)           # for generalized boosted models
library(ggplot2)       # for autoplot() function
library(pdp)           # for partial dependence plots
library(randomForest)  # for classic random forest
library(ranger)        # a much faster implementation of random forest
library(rpart)         # for binary recursive partitioning
library(rpart.plot)    # for plotting tree diagrams
library(vip)           # for variable importance plots
library(xgboost)       # for eXtreme Gradient Boosting

# Mushroom example -------------------------------------------------------------

# Load the mushroom data
path <- paste0("https://raw.githubusercontent.com/bgreenwell/",
               "MLDay18/master/data/mushroom.csv")
mushroom <- read.csv(path)  # load the data from GitHub
mushroom$veil.type <- NULL  # only takes on a single value  

# Partition the data into train/test sets
set.seed(101)
trn_id <- createDataPartition(
  y = mushroom$Edibility, p = 0.5, list = FALSE
)
trn <- mushroom[trn_id, ]   # training data
tst <- mushroom[-trn_id, ]  # test data

# Function to calculate accuracy
accuracy <- function(pred, obs) {
  sum(diag(table(pred, obs))) / length(obs)
}

# Decision stump (test error = 1.53%):
cart1 <- rpart(
  Edibility ~ ., data = trn,
  control = rpart.control(maxdepth = 1) 
)

# Get test set predictions
pred1 <- predict(
  cart1, newdata = tst, 
  type = "class"
)

# Compute test set accuracy
accuracy(
  pred = pred1, 
  obs = tst$Edibility
)

# Optimal tree (test error = 0%):
cart2 <- rpart(
  Edibility ~ ., data = trn, 
  control = list(cp = 0, minbucket = 1, minsplit = 1) 
)

# Get test set predictions
pred2 <- predict(
  cart2, newdata = tst, 
  type = "class"
)

# Compute test set accuracy
accuracy(
  pred = pred2, 
  obs = tst$Edibility
)

# Test set confusion matrices
confusionMatrix(pred1, tst$Edibility)
confusionMatrix(pred2, tst$Edibility)

# Tree diagram (deep tree)
prp(cart1,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[cart1$frame$yval])

# Tree diagram (shallow tree)
prp(cart2,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[cart2$frame$yval])


# Your turn --------------------------------------------------------------------


# Spam example -----------------------------------------------------------------

# Load the data
data(spam, package = "kernlab")

# Partition the data into train/test sets
set.seed(101)  # for reproducibility
trn_id <- createDataPartition(spam$type, p = 0.7, list = FALSE)
trn <- spam[trn_id, ]                # training data
tst <- spam[-trn_id, ]               # test data
xtrn <- subset(trn, select = -type)  # training data features
xtst <- subset(tst, select = -type)  # test data features
ytrn <- trn$type                     # training data response

# Fit a classification tree (cp found using k-fold CV)
spam_tree <- rpart(type ~ ., data = trn, cp = 0.001) 
pred <- predict(spam_tree, newdata = xtst, type = "class")

# Compute test set accuracy
(spam_tree_acc <- accuracy(pred = pred, obs = tst$type))

# Tree diagram
prp(spam_tree,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[spam_tree$frame$yval])

# Extract tibble of variable importance scores
vip::vi(spam_tree)  

# Construct ggplot2-based variable importance plot
vip(spam_tree, num_features = 10)  


# Simulated sine wave data -----------------------------------------------------

# Simulate some sine wave data
set.seed(1112)  # for reproducibility
x <- seq(from = 0, to = 2 * pi, length = 100)
y <- sin(x) + rnorm(length(x), sd = 0.5)
plot(x, y)
lines(x, sin(x))
legend("topright", legend = "True function", lty = 1, inset = 0.01,
       box.col = "transparent")

# Fit a single regression tree
fit <- rpart(y ~ x, cp = 0)
pred <- predict(fit)
plot(x, y)
lines(x, sin(x))
cols <- RColorBrewer::brewer.pal(9, "Set1")
lines(x, pred, col = cols[1L], lwd = 2)
lgnd <- c("True function", "Single tree")
legend("topright", legend = lgnd, col = c("black", cols[1L]), 
       lty = 1, inset = 0.01, box.col = "transparent")

# Fit many regression trees to bootstrap samples
plot(x, y)
nsim <- 1000
pred_mat <- matrix(nrow = length(x), ncol = nsim)
set.seed(1145)  # for reproducibility
id <- replicate(nsim, sort(sample(length(x), replace = TRUE)))
for (i in 1:nsim) {
  fit <- rpart(y[id[, i]] ~ x[id[, i]], cp = 0)
  pred_mat[, i] <- predict(fit)
  lines(x[id[, i]], pred_mat[, i], 
        col = adjustcolor(cols[2L], alpha.f = 0.05))
}
lines(x, sin(x))
lines(x, pred, col = cols[1L], lwd = 2)
lgnd <- c("True function", "Single tree", "Bootstrapped tree")
legend("topright", legend = lgnd, col = c("black", cols[1L:2L]), 
       lty = 1, inset = 0.01, box.col = "transparent")

# Average results
plot(x, y)
for (i in 1:nsim) {
  lines(x[id[, i]], pred_mat[, i], 
        col = adjustcolor(cols[2L], alpha.f = 0.05))
}
lines(x, sin(x))
lines(x, pred, col = cols[1L], lwd = 2)
lines(x, apply(pred_mat, MARGIN = 1, FUN = mean), col = cols[6L], lwd = 2)
lgnd <- c("True function", "Single tree", "Bootstrapped tree", "Averaged trees")
legend("topright", legend = lgnd, col = c("black", cols[c(1, 2, 6)]), lty = 1, 
       inset = 0.01, box.col = "transparent")


# Circle in square example revisited -------------------------------------------

# Simulate data
set.seed(1432)
circle <- as.data.frame(mlbench::mlbench.circle(
  n = 200,
  d = 2
))
names(circle) <- c("x1", "x2", "y")  # rename columns

# Plot decision boundary: CART
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5)
)

# Fit tree-based models
fit_cart <- rpart(y ~ ., data = circle)
set.seed(1651)  # for reproducibility
fit_rf <- randomForest(y ~ ., data = circle, ntree = 1000)

# Grid over which to evaluate decision boundaries
npts <- 500
xgrid <- expand.grid(
  x1 = seq(from = -1.25, 1.25, length = npts),
  x2 = seq(from = -1.25, 1.25, length = npts)
)

# Predicted probabilities
prob_cart <- predict(fit_cart, newdata = xgrid, type = "prob")
prob_rf <- predict(fit_rf, newdata = xgrid, type = "prob")

# Setup for 1-by-2 grid of plots
par(mfrow = c(1, 2))

# Plot decision boundary: CART
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "Single tree"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_cart[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: random forest
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "1000 trees"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_rf[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)


# Random forests ---------------------------------------------------------------

# Fit a bagger model
set.seed(1633)  # for reproducibility
spam_bag <- randomForest(
  type ~ ., 
  data = trn, 
  ntree = 250,
  mtry = ncol(xtrn),  # use all available features 
  xtest = subset(tst, select = -type),
  ytest = tst$type,
  keep.forest = TRUE
)

# Compute test error
pred <- predict(spam_bag, newdata = xtst, type = "class")
spam_bag_acc <- accuracy(pred = pred, obs = tst$type)

# Plot test error
dark2 <- RColorBrewer::brewer.pal(8, "Dark2")
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_bag$ntree), spam_bag$test$err.rate[, "Test"], type = "l", 
     col = dark2[1L], ylim = c(0.04, 0.11), las = 1,
     ylab = "Test error", xlab = "Number of trees")
abline(h = 1 - spam_tree_acc, lty = 2, col = "black")
abline(h = 1 - spam_bag_acc, lty = 2, col = dark2[1L])
legend("topright", c("Single tree", "Bagging"),
       col = c("black", dark2[1L]), lty = c(2, 1), lwd = 1)

# Fit a random forest
set.seed(1633)  # for reproducibility
spam_rf <- randomForest(
  type ~ ., 
  data = trn, 
  ntree = 250,
  mtry = 7,  # floor(sqrt(p))  
  xtest = subset(tst, select = -type),
  ytest = tst$type,
  keep.forest = TRUE
)

# Compute test error
pred <- predict(spam_rf, newdata = xtst, type = "class")
spam_rf_acc <- accuracy(pred = pred, obs = tst$type)

# Plot test error
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_rf$ntree), spam_rf$test$err.rate[, "Test"], type = "l", 
     col = dark2[4L], ylim = c(0.04, 0.11), 
     ylab = "Test error", xlab = "Number of trees")
lines(seq_len(spam_rf$ntree), spam_bag$test$err.rate[, "Test"], col = dark2[1L])
abline(h = 1 - spam_tree_acc, lty = 2, col = "black")
abline(h = 1 - spam_bag_acc, lty = 2, col = dark2[1L])
abline(h = 1 - spam_rf_acc, lty = 2, col = dark2[4L])
legend("topright", c("Single tree", "Bagging", "Random forest"),
       col = c("black", dark2[c(1, 4)]), lty = c(2, 1, 1), lwd = 1)

# Bootstrap
N <- 100000
set.seed(1537)  # for reproducibility
x <- rnorm(N)
mean(x %in% sample(x, replace = TRUE))  # non-OOB proportion

# Compute test error
pred <- predict(spam_rf, newdata = xtst, type = "class")
spam_rf_acc <- accuracy(pred = pred, obs = tst$type)

# Plot test error
par(mar = c(4, 4, 0.1, 0.1))
plot(seq_len(spam_rf$ntree), spam_rf$test$err.rate[, "Test"], type = "l", 
     col = dark2[4L], ylim = c(0.04, 0.11), 
     ylab = "Error estimate", xlab = "Number of trees")
lines(seq_len(spam_rf$ntree), spam_rf$err.rate[, "OOB"], type = "l", 
     col = dark2[1L])
abline(h = spam_rf$err.rate[spam_rf$ntree, "OOB"], lty = 2, col = dark2[1L])
abline(h = 1 - spam_rf_acc, lty = 2, col = dark2[4L])
legend("topright", c("Random forest (OOB)", "Random forest (test)"),
       col = c(dark2[c(1, 4)]), lty = c(1, 1))

# Load the (corrected) Boston housing data
data(boston, package = "pdp")

# Using the randomForest package
set.seed(2007)  # for reproducibility
system.time(
  boston_rf <- randomForest(
    cmedv ~ ., data = boston, 
    ntree = 5000,
    mtry = 5,
    importance = FALSE
  )
)
boston_rf$rsq[boston_rf$ntree]

# Using the ranger package
set.seed(1652)  # for reproducibility
system.time(
  boston_ranger <- ranger(
    cmedv ~ ., data = boston, 
    num.trees = 5000, 
    mtry = 5,  # :/  
    importance = "impurity"
  )
)
boston_ranger$r.squared

# Refit models with less trees
set.seed(1453)  # for reproducibility
boston_rf <- randomForest(cmedv ~ ., data = boston, ntree = 500,
                          importance = TRUE, proximity = TRUE)
boston_ranger <- ranger(cmedv ~ ., data = boston, num.trees = 500,
                        importance = "impurity")

# Construct variable importance plots (the old way)
par(mfrow = c(1, 2))  # side-by-side plots
varImpPlot(boston_rf)  # randomForest::varImpPlot()  

# Construct variable importance plots
p1 <- vip(boston_rf, type = 1) + ggtitle("randomForest")
p2 <- vip(boston_rf, type = 2) + ggtitle("randomForest")
p3 <- vip(boston_ranger) + ggtitle("ranger")
grid.arrange(p1, p2, p3, ncol = 3)  

# PDPs for the top two predictors
p1 <- partial(boston_ranger, pred.var = "lstat", plot = TRUE)
p2 <- partial(boston_ranger, pred.var = "rm", plot = TRUE)
p3 <- partial(boston_ranger, pred.var = c("lstat", "rm"),  
              chull = TRUE, plot = TRUE)                   
grid.arrange(p1, p2, p3, ncol = 3)

# 3-D plots
pd <- attr(p3, "partial.data")  # no need to recalculate 
p1 <- plotPartial(pd, 
  levelplot = FALSE, drape = TRUE, colorkey = FALSE,
  screen = list(z = -20, x = -60)
)

# Using ggplot2
p2 <- autoplot(pd)

# ICE and c-ICE curves
p3 <- boston_ranger %>%  # %>% is automatically imported!
  partial(pred.var = "rm", ice = TRUE, center = TRUE) %>%
  autoplot(alpha = 0.1)

# Display all three plots side-by-side
grid.arrange(p1, p2, p3, ncol = 3)

# Tree diagram (shallow tree)
prp(cart2,
    type = 4,
    clip.right.labs = FALSE, 
    branch.lwd = 2,
    extra = 1, 
    under = TRUE,
    under.cex = 1.5,
    split.cex = 1.5,
    box.col = c("palegreen3", "pink")[cart2$frame$yval])

# Load the data
data(banknote, package = "alr3")

# Fit a random forest
set.seed(1701)  # for reproducibility
banknote_rf <- randomForest(
  as.factor(Y) ~ ., 
  data = banknote, 
  proximity = TRUE  
)

# Print the OOB confusion matrix
banknote_rf$confusion

# Heatmap of proximity-based distance matrix
heatmap(1 - banknote_rf$proximity, col = viridis::plasma(256))

# Dot chart of proximity-based outlier scores
outlyingness <- tibble::tibble(
  "out" = outlier(banknote_rf),  
  "obs" = seq_along(out),
  "class" = as.factor(banknote$Y)
)
ggplot(outlyingness, aes(x = obs, y = out)) +
  geom_point(aes(color = class, size = out), alpha = 0.5) +
  geom_hline(yintercept = 10, linetype = 2) +  
  labs(x = "Observation", y = "Outlyingness") +
  theme_light() +
  theme(legend.position = "none")

# Multi-dimensional scaling plot of proximity matrix
MDSplot(banknote_rf, fac = as.factor(banknote$Y), k = 2, cex = 1.5)


# Your turn --------------------------------------------------------------------


# Stochastic gradient boosting ------------------------------------------------

# Fit a GBM to the Boston housing data
set.seed(1053)  # for reproducibility
boston_gbm <- gbm(
  cmedv ~ ., 
  data = boston, 
  var.monotone = NULL,        
  distribution = "gaussian",  # "benoulli", "coxph", etc. 
  n.trees = 10000,            
  interaction.depth = 5,      
  n.minobsinnode = 10,        
  shrinkage = 0.005,          
  bag.fraction = 1,           
  train.fraction = 1,         
  cv.folds = 10  # k-fold CV often gives the best results
)

# "Best" number of trees
best_iter <- gbm.perf(
  boston_gbm, 
  method = "cv"  # or "OOB" or "test" 
)

# randomForest() PDP timing
system.time(
  pd1 <- partial(
    boston_gbm, 
    pred.var = c("lon", "nox"),
    recursive = FALSE,  
    chull = TRUE, 
    n.trees = best_iter  
  )
)

# gbm() PDP timing
system.time(
  pd2 <- partial(
    boston_gbm, 
    pred.var = c("lon", "nox"),
    recursive = TRUE,  
    chull = TRUE, 
    n.trees = best_iter  
  )
)

# Display plots side-by-side
grid.arrange(autoplot(pd1), autoplot(pd2), ncol = 2)

# Cool plot
ames <- AmesHousing::make_ames()
ggplot(ames, aes(x = Sale_Price, y = Overall_Qual)) + 
  ggridges::geom_density_ridges(aes(fill = Overall_Qual)) +  
  scale_x_continuous(labels = scales::dollar) +
  labs(x = "Sale price", y = "Overall quality") +
  theme_light() + theme(legend.position = "none")

# Construct data set
ames <- AmesHousing::make_ames()

# Feature matrix  # or xgb.DMatrix or sparse matrix  
X <- data.matrix(subset(ames, select = -Sale_Price))

# Use k-fold cross-validation to find the "optimal" number of trees
set.seed(1214)  # for reproducibility
ames_xgb_cv <- xgb.cv(
  data = X, 
  label = ames$Sale_Price, 
  objective = "reg:linear",
  nrounds = 10000, 
  max_depth = 5, 
  eta = 0.01, 
  subsample = 1,          
  colsample = 1,          
  num_parallel_tree = 1,  
  eval_metric = "rmse",   
  early_stopping_rounds = 50,
  verbose = 0,
  nfold = 5
)

# Plot cross-validation results
plot(test_rmse_mean ~ iter, data = ames_xgb_cv$evaluation_log, type = "l", 
     ylim = c(0, 200000), xlab = "Number of trees", ylab = "RMSE",
     main = "Results from using xgb.cv()")
lines(train_rmse_mean ~ iter, data = ames_xgb_cv$evaluation_log, col = "red2")
abline(v = ames_xgb_cv$best_iteration, lty = 2)
legend("topright", legend = c("Train", "CV"), lty = 1, col = c("red2", 1),
       inset = 0.15)

# Fit an XGBoost model
set.seed(203)  # for reproducibility
ames_xgb <- xgboost(         # tune using `xgb.cv()`
  data = X, 
  label = ames$Sale_Price, 
  objective = "reg:linear",  # loss function 
  nrounds = 2771,            # number of trees  
  max_depth = 5,             # interaction depth  
  eta = 0.01,                # learning rate  
  subsample = 1,             
  colsample = 1,             
  num_parallel_tree = 1,     
  eval_metric = "rmse",      
  verbose = 0,
  save_period = NULL         
)

# Variable importance plots
p1 <- vip(ames_xgb, feature_names = colnames(X), type = "Gain")
p2 <- vip(ames_xgb, feature_names = colnames(X), type = "Cover")
p3 <- vip(ames_xgb, feature_names = colnames(X), type = "Frequency")
grid.arrange(p1, p2, p3, ncol = 3)

# By default, `vip()` plots the top 10 features
vip(ames_xgb, feature_names = colnames(X), type = "Gain", 
    num_features = nrow(X), bar = FALSE)

# Partial dependence plots
oq_ice <- partial(ames_xgb, pred.var = "Overall_Qual", ice = TRUE, 
                  center = TRUE, train = X)
p4 <- autoplot(partial(ames_xgb, pred.var = "Gr_Liv_Area", train = X))
p5 <- autoplot(partial(ames_xgb, pred.var = "Garage_Cars", train = X))
p6 <- autoplot(oq_ice, alpha = 0.1)
grid.arrange(p4, p5, p6, ncol = 3)

# Partial dependence plots for the top/bottom three features
ames_vi <- vi(ames_xgb, feature_names = colnames(X), type = "Gain")
feats <- c(head(ames_vi, n = 3)$Variable, tail(ames_vi, n = 3)$Variable)
pds <- lapply(feats, FUN = function(x) {
  pd <- cbind(x, partial(ames_xgb, pred.var = x, train = X))
  names(pd) <- c("xvar", "xval", "yhat")
  pd
})
pds <- do.call(rbind, pds)
ggplot(pds, aes(x = xval, y = yhat)) +
  geom_line(size = 1.5) +
  geom_hline(yintercept = mean(ames$Sale_Price), linetype = 2, col = "red2") +
  facet_wrap( ~ xvar, scales = "free_x") +
  labs(x = "", y = "Partial dependence") +
  theme_light()
