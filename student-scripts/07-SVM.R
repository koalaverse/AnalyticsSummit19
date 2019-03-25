# Setup ------------------------------------------------------------------------

# Colors
dark2 <- RColorBrewer::brewer.pal(8, "Dark2")
set1 <- RColorBrewer::brewer.pal(9, "Set1")

# Plotting function modified from svmpath::svmpath()
plot_svmpath <- function(x, step = max(x$Step)) {
  object = x
  f = predict(object, lambda = object$lambda[step], type = "function")
  x = object$x
  y = object$y
  Elbow = object$Elbow[[step]]
  alpha = object$alpha[, step]
  alpha0 = object$alpha0[step]
  lambda = object$lambda[step]
  x <- x[, 1:2]
  plotargs = list(x = x[, ], type = "n", xlab = "Income (standardized)", 
                  ylab = "Lot size (standardized)", main = "")
  do.call("plot", plotargs)
  dark2 <- RColorBrewer::brewer.pal(8, "Dark2")
  points(x, cex = 1.2, pch = c(17, 19)[ifelse(norm2d$y == 1, 1, 2)],
         col = adjustcolor(dark2[ifelse(norm2d$y == 1, 1, 2)], alpha.f = 0.5))
  beta <- (alpha * y) %*% x
  abline(-alpha0/beta[2], -beta[1]/beta[2], col = "black", lwd = 1)
  abline(lambda/beta[2] - alpha0/beta[2], -beta[1]/beta[2], 
         col = "black", lwd = 1, lty = 2)
  abline(-lambda/beta[2] - alpha0/beta[2], -beta[1]/beta[2], 
         col = "black", lwd = 1, lty = 2)
  points(x[Elbow, ], pch = 19, cex = 1.2)
}


# Introduction -----------------------------------------------------------------

# Setup for 1-by-2 grid of plots
par(mfrow = c(1, 2))

# Scatterplot of overlapping data
set.seed(101)
df <- as.data.frame(mlbench::mlbench.2dnormals(100))
plot(
  x.2 ~ x.1,
  data = df,
  cex = 1.2,
  pch = c(17, 19)[df$classes],
  col = adjustcolor(dark2[df$classes], alpha.f = 0.5),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Overlapping classes"
)

# Scatterplot of data with nonlinear decision boundary
set.seed(102)
df <- as.data.frame(mlbench::mlbench.spirals(300, cycles = 2, sd = 0.09))
plot(
  x.2 ~ x.1,
  data = df,
  cex = 1.2,
  pch = c(17, 19)[df$classes],
  col = adjustcolor(dark2[df$classes], alpha.f = 0.5),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Nonlinear decision boundary"
)


# Hard margin classifier/optimal separating hyperplane -------------------------

# Load required packages
library(kernlab)  # for fitting support vector machines
library(MASS)     # for LDA/QDA

# Simulate data
set.seed(805)
norm2d <- as.data.frame(mlbench::mlbench.2dnormals(
  n = 100,
  cl = 2,
  r = 4,
  sd = 1
))
names(norm2d) <- c("x1", "x2", "y")  # rename columns


# Scatterplot
dev.off()  # reset graphics device
plot(
  formula = x2 ~ x1, 
  data = norm2d, 
  cex = 1.2,
  pch = c(17, 19)[norm2d$y],
  col = adjustcolor(dark2[norm2d$y], alpha.f = 0.5),
  xlab = "Income (standardized)",
  ylab = "Lot size (standardized)"
)

# Fit a Logistic regression, linear discriminant analysis (LDA), and optimal
# separating hyperplane (OSH). Note: we sometimes refer to the OSH as the hard 
# margin classifier (HMC)
fit_glm <- glm(as.factor(y) ~ ., data = norm2d, family = binomial)
fit_lda <- lda(as.factor(y) ~ ., data = norm2d)
invisible(capture.output(fit_hmc <- ksvm(  # use ksvm() to find the OSH
  x = data.matrix(norm2d[c("x1", "x2")]),
  y = as.factor(norm2d$y), 
  kernel = "vanilladot",  # no fancy kernel, just ordinary dot product
  C = Inf,                # to approximate hard margin classifier
  prob.model = TRUE       # needed to obtain predicted probabilities
)))

# Grid over which to evaluate decision boundaries
npts <- 500
xgrid <- expand.grid(
  x1 = seq(from = -6, 6, length = npts),
  x2 = seq(from = -6, 6, length = npts)
)

# Predicted probabilities (as a two-column matrix)
prob_glm <- predict(fit_glm, newdata = xgrid, type = "response")
prob_glm <- cbind("1" = 1 - prob_glm, "2" = prob_glm)
prob_lda <- predict(fit_lda, newdata = xgrid)$posterior
prob_hmc <- predict(fit_hmc, newdata = xgrid, type = "probabilities")

# Scatterplot
plot(
  formula = x2 ~ x1, 
  data = norm2d, 
  asp = 1,
  cex = 1.2,
  pch = c(17, 19)[norm2d$y],
  col = adjustcolor(dark2[norm2d$y], alpha.f = 0.5),
  xlab = "Income (standardized)",
  ylab = "Lot size (standardized)"
)

# Plot decision boundary: logistic regression
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_glm[, 1L], nrow = npts), 
  levels = 0.5,
  col = set1[1L], 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: LDA
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_lda[, 1L], nrow = npts), 
  levels = 0.5,
  col = set1[2L], 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: OSH
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_hmc[, 1L], nrow = npts), 
  levels = 0.5,
  col = set1[3L], 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Add plot legend
legend(
  x = "topleft",
  legend = c("Logistic regression", "LDA", "HMC"),
  col = set1[1L:3L],
  lty = 1,
  inset = 0.02,
  bty = "n",
  cex = 0.5
)


# Manually finding the OSH -----------------------------------------------------

# Scatterplot
plot(
  formula = x2 ~ x1,
  data = norm2d, 
  asp = 1,
  cex = 1.2,
  pch = c(17, 19)[norm2d$y],
  col = adjustcolor(dark2[norm2d$y], alpha.f = 0.5),
  xlab = "Income (standardized)",
  ylab = "Lot size (standardized)"
)

# Plot convex hull for each class
hpts1 <- chull(norm2d[norm2d$y == 1, c("x1", "x2")])
hpts1 <- c(hpts1, hpts1[1L])
hpts2 <- chull(norm2d[norm2d$y == 2, c("x1", "x2")])
hpts2 <- c(hpts2, hpts2[1L])
lines(norm2d[norm2d$y == 1, c("x1", "x2")][hpts1, c("x1", "x2")])
lines(norm2d[norm2d$y == 2, c("x1", "x2")][hpts2, c("x1", "x2")])

# Identify the support vectors; that is, the training data on the convex hull
# that are the closest between the two classes
sv <- norm2d[fit_osh@alphaindex[[1L]], c("x1", "x2")]  # 16-th and 97-th observations
points(sv, pch = 19, cex = 1.2)

# Add joining line segment
arrows(sv[1L, 1L], sv[1L, 2L], sv[2L, 1L], sv[2L, 2L], code = 3, length = 0.1)

# Plot the OSH; that is, the perpendicular bisector of the line segment 
# joining the two support vectors
slope <- -1 / ((sv[2L, 2L] - sv[1L, 2L]) / (sv[2L, 1L] - sv[1L, 1L]))
midpoint <- apply(sv, 2, mean)
abline(
  a = -slope * midpoint[1L] + midpoint[2L], 
  b = slope
)

# Plot margin boundaries
abline(
  a = -slope * sv[1L, 1L] + sv[1L, 2L], 
  b = slope,
  lty = 2
)
abline(
  a = -slope * sv[2L, 1L] + sv[2L, 2L], 
  b = slope,
  lty = 2
)

# Label margin
pBrackets::brackets(
  x2 = sv[1L, 1L]+0.5, 
  y2 = sv[1L, 2L]-0.5,
  x1 = midpoint[1L]+0.5,
  y1 = midpoint[2L]-0.5,
  type = 1,
  col = set1[1L],
  lwd = 2
)
text(1.5, -2.25, label = "M", col = set1[1L], srt = 40)


# Soft margin classifier -------------------------------------------------------

# Add an outlier
norm2d <- rbind(norm2d, data.frame("x1" = 0.5, "x2" = 1, "y" = 2))

# Scatterplot
plot(
  formula = x2 ~ x1, 
  data = norm2d, 
  asp = 1,
  cex = 1.2,
  pch = c(17, 19)[norm2d$y],
  col = adjustcolor(dark2[norm2d$y], alpha.f = 0.5),
  xlab = "Income (standardized)",
  ylab = "Lot size (standardized)"
)

# Add an arrow pointing to the outlier
arrows(-1, 4, 0.5, 1)
text(-1, 4, label = "Outlier?", pos = 3)

# Fit a Logistic regression, linear discriminant analysis (LDA), and optimal
# separating hyperplane (OSH)
#
# Note: we sometimes refer to the OSH as the hard margin classifier
fit_glm <- glm(as.factor(y) ~ ., data = norm2d, family = binomial)
fit_lda <- lda(as.factor(y) ~ ., data = norm2d)
fit_hmc <- ksvm(  # use ksvm() to find the OSH
  x = data.matrix(norm2d[c("x1", "x2")]),
  y = as.factor(norm2d$y), 
  kernel = "vanilladot",  # no fancy kernel, just ordinary dot product
  C = Inf,                # to approximate maximal margin classifier
  prob.model = TRUE       # needed to obtain predicted probabilities
)

# Grid over which to evaluate decision boundaries
npts <- 500
xgrid <- expand.grid(
  x1 = seq(from = -6, 6, length = npts),
  x2 = seq(from = -6, 6, length = npts)
)

# Predicted probabilities (as a two-column matrix)
prob_glm <- predict(fit_glm, newdata = xgrid, type = "response")
prob_glm <- cbind("1" = 1 - prob_glm, "2" = prob_glm)
prob_lda <- predict(fit_lda, newdata = xgrid)$posterior
prob_hmc <- predict(fit_hmc, newdata = xgrid, type = "probabilities")

# Scatterplot
plot(
  formula = x2 ~ x1, 
  data = norm2d, 
  asp = 1,
  cex = 1.2,
  pch = c(17, 19)[norm2d$y],
  col = adjustcolor(dark2[norm2d$y], alpha.f = 0.5),
  xlab = "Income (standardized)",
  ylab = "Lot size (standardized)"
)

# Plot decision boundary: logistic regression
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_glm[, 1L], nrow = npts), 
  levels = 0.5,
  col = set1[1L], 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: LDA
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_lda[, 1L], nrow = npts), 
  levels = 0.5,
  col = set1[2L], 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: OSH
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_hmc[, 1L], nrow = npts), 
  levels = 0.5,
  col = set1[3L], 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Add plot legend
legend(
  x = "topleft",
  legend = c("Logistic regression", "LDA", "HMC"),
  col = set1[1L:3L],
  lty = 1,
  inset = 0.02,
  bty = "n",
  cex = 0.5
)


# Fitting the entire egularization path ----------------------------------------

# Load required packages
library(svmpath)

# Fit the entire regularization path
fit_smc <- svmpath(
  x = data.matrix(norm2d[c("x1", "x2")]), 
  y = ifelse(norm2d$y == 1, 1, -1)
)

# Plot both extremes
par(mfrow = c(1, 2))
plot_svmpath(fit_smc, step = min(fit_smc$Step))
plot_svmpath(fit_smc, step = max(fit_smc$Step))

# Make a GIF of all the possible steps; requires ImageMagick to be installed!
# png(file = "GIFs/svmpath%02d.png", width = 4, height = 4, 
#     units = "in", res = 300)
# for (i in sort(unique(fit_smc$Step))) {
#   plot_svmpath(fit_smc, step = i)
# }
# dev.off()
# sys_string <- paste(
#   "convert -delay 10",
#   paste("GIFs/svmpath", sprintf('%0.2d', max(fit_smc$Step)), ".png", 
#         sep = "", collapse = " "),
#   "GIFs/svmpath.gif"
# )
# system(sys_string)
# file.remove(list.files(path = "GIFs", pattern = ".png", full.names = TRUE))


# Support vector machines ------------------------------------------------------

# Simulate data
set.seed(1432)
circle <- as.data.frame(mlbench::mlbench.circle(
  n = 200,
  d = 2
))
names(circle) <- c("x1", "x2", "y")  # rename columns

# Scatterplot
dev.off()  # reset graphics device
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5)
)

# Load required packages
library(plotly)

# Enlarge feature space
circle_3d <- circle
circle_3d$x3 <- circle_3d$x1^2 + circle_3d$x2^2

# 3-D scatterplot (enlarged feature space)
plot_ly(
  circle_3d, 
  x = ~x1, 
  y = ~x2, 
  z = ~x3, 
  color = ~y, 
  colors = dark2[1L:2L]
) %>%
  add_markers()

# Fit a Logistic regression, quadratic discriminant analysis (QDA), and a 
# support vector machine (SVM)
fit_glm <- glm(as.factor(y) ~ x1 + x1 + I(x1^2) + I(x2^2), data = circle, 
               family = binomial)
fit_qda <- qda(as.factor(y) ~ ., data = circle)
fit_svm_poly <- ksvm( 
  x = data.matrix(circle[c("x1", "x2")]),
  y = as.factor(circle$y), 
  kernel = "polydot",       # polynomial kernel
  kpar = list(degree = 2),  # kernel parameters
  C = Inf,                  # to approximate maximal margin classifier
  prob.model = TRUE         # needed to obtain predicted probabilities
)
fit_svm_rbf <- ksvm( 
  x = data.matrix(circle[c("x1", "x2")]),
  y = as.factor(circle$y), 
  kernel = "rbfdot",        # polynomial kernel
  C = Inf,                  # to approximate maximal margin classifier
  prob.model = TRUE         # needed to obtain predicted probabilities
)

# Grid over which to evaluate decision boundaries
npts <- 500
xgrid <- expand.grid(
  x1 = seq(from = -1.25, 1.25, length = npts),
  x2 = seq(from = -1.25, 1.25, length = npts)
)

# Predicted probabilities (as a two-column matrix)
prob_glm <- predict(fit_glm, newdata = xgrid, type = "response")
prob_glm <- cbind("1" = 1 - prob_glm, "2" = prob_glm)
prob_qda <- predict(fit_qda, newdata = xgrid)$posterior
prob_svm_poly <- predict(fit_svm_poly, newdata = xgrid, type = "probabilities")
prob_svm_rbf <- predict(fit_svm_rbf, newdata = xgrid, type = "probabilities")

# Setup for 2-by-2 grid of plots
par(mfrow = c(2, 2))

# Plot decision boundary: logistic regression
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "Polynomial logistic regression"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_glm[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: QDA
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "QDA"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_qda[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: SVM (poly)
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "SVM: polynomial kernel"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_svm_poly[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: SVM (RBF)
plot(
  formula = x2 ~ x1, 
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "SVM: RBF kernel"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_svm_rbf[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)

# Plot decision boundary: MARS (logit)
library(earth)
fit_mars <- earth(as.factor(y) ~ x1 + x2, data = circle,
                  glm = list(family = binomial))
prob_mars <- predict(fit_mars, newdata = xgrid, type = "response")
dev.off()  # reset graphics device
plot(
  formula = x2 ~ x1,
  data = circle, 
  cex = 1.2,
  pch = c(17, 19)[circle$y],
  col = adjustcolor(dark2[circle$y], alpha.f = 0.5),
  main = "MARS: logit"
)
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(prob_mars[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 2,
  lty = 1,
  add = TRUE
)


# Two spirals benchmark problem -------------------------------------------------------------

# Load required packages
library(kernlab)  # for fitting SVMs
library(mlbench)  # for ML benchmark data sets  

# Simulate train and test sets
set.seed(0841)
trn <- as.data.frame(
  mlbench.spirals(300, cycles = 2, sd = 0.09)
)
tst <- as.data.frame(
  mlbench.spirals(10000, cycles = 2, sd = 0.09)
)
names(trn) <- names(tst) <- c("x1", "x2", "classes")

# Plot training data
plot(
  x2 ~ x1,
  data = trn,
  cex = 1.2,
  pch = c(17, 19)[trn$classes],
  col = adjustcolor(dark2[trn$classes], alpha.f = 0.5),
  xlab = expression(X[1]),
  ylab = expression(X[2])
)

# Fit an SVM using a radial basis function kernel
spirals_rbf <- ksvm(
  classes ~ x1 + x2, data = trn, 
  kernel = "rbfdot",  
  C = 500,  # I just picked a value     
  prob.model = TRUE   
)

# Grid over which to evaluate decision boundaries
npts <- 500
xgrid <- expand.grid(
  x1 = seq(from = -2, 2, length = npts),
  x2 = seq(from = -2, 2, length = npts)
)

# Predicted probabilities (as a two-column matrix)
spirals_rbf_prob <- 
  predict(spirals_rbf, newdata = xgrid, type = "probabilities")

# Plot decision boundary
contour(
  x = sort(unique(xgrid$x1)), 
  y = sort(unique(xgrid$x2)), 
  z = matrix(spirals_rbf_prob[, 1L], nrow = npts), 
  levels = 0.5,
  col = "black", 
  drawlabels = FALSE, 
  lwd = 1,
  lty = 1,
  add = TRUE
)

# Test set confusion matrix
(tab <- table(
  pred = predict(spirals_rbf, newdata = tst),  # predicted outcome
  obs = tst$classes                    # observed outcome
))

# Test set error
1 - sum(diag(tab)) / nrow(tst)  # test error ~ 8.44%


# SVM tuning parameters --------------------------------------------------------

# Linear (i.e., ordinary inner product)
caret::getModelInfo("svmLinear")$svmLinear$parameters

# Polynomial kernel
caret::getModelInfo("svmPoly")$svmPoly$parameters

# Radial basis kernel
caret::getModelInfo("svmRadial")$svmRadial$parameters


# Job attrition example -----------------------------------------------------------

# Load required packages
library(caret)    # for classification and regression training
library(dplyr)    # for data wrangling
library(ggplot2)  # for more awesome plotting
library(rsample)  # for attrition data and data splitting
library(pdp)      # for PDPs

# Same setup as Naive Bayes module!  

# Load the attrition data
attrition <- attrition %>%
  mutate(  # convert some numeric features to factors
    JobLevel = factor(JobLevel),
    StockOptionLevel = factor(StockOptionLevel),
    TrainingTimesLastYear = factor(TrainingTimesLastYear)
  )

# Train and test splits
set.seed(123)  # for reproducibility
split <- initial_split(attrition, prop = 0.7, strata = "Attrition")
trn <- training(split)
tst <- testing(split)

# Control params for SVM
ctrl <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  
)

# Tune an SVM
set.seed(1854)  # for reproducibility
attr_svm <- train(
  Attrition ~ ., 
  data = trn,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",                     
  trControl = ctrl,
  tuneGrid = data.frame(
    sigma = 0.008071434, 
    C = seq(from = 0.1, to = 5, length = 30)
  )
)

# Plot tuning results
ggplot(attr_svm) + theme_light()

# Filter-based variable importance plot
plot(varImp(attr_svm))

# Filter-based variable 
# importance scores
plot(varImp(attr_svm))

# Partial dependence plots
features <- c(
  "MonthlyIncome", 
  "TotalWorkingYears", 
  "OverTime", 
  "YearsAtCompany"
)
pdfs <- lapply(features, function(x) {
  autoplot(partial(attr_svm, pred.var = x, prob = TRUE)) +
    theme_light()
})
grid.arrange(
  grobs = pdfs,  
  ncol = 2
)

# Load required packages
library(pROC)  

# Plot train and test ROC curves
roc_trn <- roc(  # train AUC: 0.9717
  predictor = predict(attr_svm, newdata = trn, type = "prob")$Yes, 
  response = trn$Attrition,
  levels = rev(levels(trn$Attrition))
)
roc_tst <- roc(  # test AUC: 0.8567
  predictor = predict(attr_svm, newdata = tst, type = "prob")$Yes, 
  response = tst$Attrition,
  levels = rev(levels(tst$Attrition))
)
plot(roc_trn)
lines(roc_tst, col = "dodgerblue2")
legend("bottomright", legend = c("Train", "Test"), bty = "n", cex = 2.5,
       col = c("black", "dodgerblue2"), inset = 0.01, lwd = 2)


# Your turn --------------------------------------------------------------------

