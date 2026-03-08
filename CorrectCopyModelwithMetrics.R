#install.packages revert to xgboost package 1.7.11.1
#check packages before running
install.packages("remotes")
remotes::install_version("xgboost", version = "1.7.11.1")
#remotes::install_version("ggplot2", version = "3.5.2")
install.packages("caret")
#install.packages("xgboost")
install.packages("pROC")
#install.packages("ggplot2")
install.packages("dplyr")
install.packages("DT")
library(caret)
library(xgboost)
library(pROC)
library(ggplot2)

# Step 1: Load the cleaned data with strings as factors
#data <- read.csv("C:/Users/dr112507/Documents/TRXFN_Clean.csv", stringsAsFactors = TRUE)

#data <- read.csv("C:/Users/drjen/Downloads/TRXFN_Clean.csv", stringsAsFactors = TRUE, header = TRUE)
data <- read.csv("TRXFN_Clean.csv", stringsAsFactors = TRUE, header = TRUE)
model <- readRDS("xgb_units_model_cleaned8.rds")
# Step 2: Force factor for Indications (adjust column name as needed)
data$Cesarean.Indications <- factor(data$Cesarean.Indications)

# Step 3: Print mapping of Indications factor levels to integer codes
cat("Mapping of Cesarean.Indications factor levels to integer codes:\n")
ind_levels <- levels(data$Cesarean.Indications)
ind_codes <- seq_along(ind_levels)
mapping_df <- data.frame(LevelCode = ind_codes, LevelName = ind_levels)
print(mapping_df)

# Step 4: Create binary churn variable (target)
data$churn <- factor(ifelse(data$Units == 0, "no", "yes"), levels = c("no", "yes"))

# Step 5: Remove ID and Units columns to prevent leakage
data_model <- data[, !(names(data) %in% c("id", "Units"))]
str(data_model)
levels(data_model$churn)
head(data_model)
# Step 6: Train/test split 70/30 copy changed to 80/20
set.seed(123)
trainIndex <- createDataPartition(data_model$churn, p = 0.8, list = FALSE)
trainData <- data_model[trainIndex, ]
testData  <- data_model[-trainIndex, ]

# Step 7: Remove factor variables with only one level in training data
single_level_factors <- sapply(trainData, function(col) {
  is.factor(col) && length(unique(col)) < 2
})
if (any(single_level_factors)) {
  cat("Removing single-level factor variables:\n")
  print(names(trainData)[single_level_factors])
  trainData <- trainData[, !single_level_factors]
}

# Step 8: Ensure testData has same columns as trainData
testData <- testData[, names(trainData)]

# Step 9: Set up training control with cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = TRUE
)

# Step 10: Define XGBoost tuning grid
xgb_grid <- expand.grid(
  nrounds = c(50, 100),
  max_depth = c(3, 6),
  eta = c(0.1),
  gamma = c(0),
  colsample_bytree = c(0.8),
  min_child_weight = c(1),
  subsample = c(0.8)
)

# Step 11: Train the XGBoost model
set.seed(123)
xgb_model <- train(
  churn ~ .,
  data = trainData,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = xgb_grid
)



# Step 13: Predict on test data
test_probs <- predict(xgb_model, newdata = testData, type = "prob")
test_preds <- predict(xgb_model, newdata = testData)

# Step 14: Evaluate performance
conf_matrix <- confusionMatrix(test_preds, testData$churn, positive = "yes")
print(conf_matrix)

# Step 15: ROC and AUC
roc_obj <- roc(response = testData$churn, predictor = test_probs$yes)
plot.roc(roc_obj, col = "black", main = "ROC Curve", print.auc = TRUE)

# Step 16: Summarize key performance metrics
performance_df <- data.frame(
  Metric = c("AUC", "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score"),
  Value = round(c(
    auc(roc_obj),
    conf_matrix$overall["Accuracy"],
    conf_matrix$byClass["Sensitivity"],
    conf_matrix$byClass["Specificity"],
    conf_matrix$byClass["Pos Pred Value"],
    2 * (conf_matrix$byClass["Pos Pred Value"] * conf_matrix$byClass["Sensitivity"]) /
      (conf_matrix$byClass["Pos Pred Value"] + conf_matrix$byClass["Sensitivity"])
  ), 4)
)
print(performance_df)

# Step 17: Feature importance plot with ggplot2 for color control
xgb_final <- xgb_model$finalModel
importance_matrix <- xgb.importance(model = xgb_final)

importance_df <- importance_matrix[1:10, ]
ggplot(importance_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  labs(title = "Top 10 Important Features", x = "Feature", y = "Gain") +
  theme_minimal()
install.packages("precrec", dependencies = TRUE)
library(precrec)
# Load required package
library(pROC)
data_model$churn <- factor(data_model$churn, levels = c("no", "yes"))


# Predict probabilities for positive class "yes"
pred_probs <- predict(xgb_model, newdata = testData, type = "prob")[, "yes"]
data_model$churn <- factor(data_model$churn, levels = c("no", "yes"))
# Compute ROC curve
roc_obj <- roc(response = testData$churn, predictor = pred_probs, levels = c("no", "yes"), direction = "<")

# Calculate AUC explicitly with pROC namespace to avoid conflicts
auc_value <- pROC::auc(roc_obj)
cat("ROC AUC:", round(auc_value, 4), "\n")

# Plot ROC curve
plot(roc_obj,
     col = "#1f77b4",
     lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc_value, 4), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")
#PR curve
library(ggplot2)

# 1. Get predicted probabilities for positive class "yes" (already have pred_probs)

# 2. Calculate precision and recall at different thresholds
thresholds <- sort(unique(pred_probs), decreasing = TRUE)
precision <- numeric(length(thresholds))
recall <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  thresh <- thresholds[i]
  preds <- ifelse(pred_probs >= thresh, "yes", "no")  # factor levels same as churn
  preds <- factor(preds, levels = c("no", "yes"))
  
  TP <- sum(preds == "yes" & testData$churn == "yes")
  FP <- sum(preds == "yes" & testData$churn == "no")
  FN <- sum(preds == "no" & testData$churn == "yes")
  
  precision[i] <- ifelse(TP + FP == 0, 1, TP / (TP + FP))
  recall[i] <- ifelse(TP + FN == 0, 0, TP / (TP + FN))
}

# 3. Compute approximate PR-AUC (trapezoidal rule)
ord <- order(recall)
pr_auc <- sum(diff(recall[ord]) * (precision[ord][-length(precision)] + precision[ord][-1]) / 2)

# 4. Prepare data frame for plotting
pr_df <- data.frame(Recall = recall, Precision = precision)

# 5. Plot PR curve with ggplot2
ggplot(pr_df, aes(x = Recall, y = Precision)) +
  geom_line(color = "#1f77b4", size = 1.2) +
  geom_area(fill = "#1f77b4", alpha = 0.2) +
  labs(title = paste("Precision-Recall Curve (PR-AUC =", round(pr_auc, 2), ")"),
       x = "Recall",
       y = "Precision") +
  theme_minimal()
#ggplot(pr_data, aes(x = recall, y = precision)) +
# geom_line(color = "#2C7BB6", linewidth = 1.2) +
# coord_cartesian(ylim = c(.8,.92)) +
# labs(title = "Precision–Recall Curve",x = "Recall",y = "Precision") +
# theme_minimal(base_size = 14)
#plot_pr <- function(pr_data, labels = NULL) {
#  y_range <- range(pr_data$precision, na.rm = TRUE)
#  p <- ggplot(pr_data, aes(recall, precision)) +
 #   geom_line(color = "#2C7BB6", linewidth = 1.2) +
 #   coord_cartesian(ylim = y_range) +
#    theme_minimal(base_size = 14) +
 #   labs(x = "Recall", y = "Precision", title = "Precision–Recall Curve")
 # if (!is.null(labels)) p <- p + geom_hline(yintercept = mean(labels), linetype = "dashed", color = "darkblue")
 # p
#}

#plot_pr(pr_data, labels)
#saving and metrics
library(caret)
library(pROC)

# ---- Assume the model is already trained as 'xgb_model' ----
# Predict probabilities and classes
pred_probs <- predict(xgb_model, newdata = testData, type = "prob")[, "yes"]
pred_class <- predict(xgb_model, newdata = testData)

# Ensure the factor levels match
testData$churn <- factor(testData$churn, levels = c("no", "yes"))
pred_class <- factor(pred_class, levels = c("no", "yes"))

# Confusion matrix
cm <- confusionMatrix(pred_class, testData$churn, positive = "yes")

# ROC object and AUC
roc_obj <- roc(response = testData$churn, predictor = pred_probs, levels = c("no", "yes"), direction = "<")
roc_auc <- pROC::auc(roc_obj)

# Extract metrics
accuracy <- cm$overall["Accuracy"]
sensitivity <- cm$byClass["Sensitivity"]
specificity <- cm$byClass["Specificity"]
precision <- cm$byClass["Precision"]
f1_score <- cm$byClass["F1"]

# Optional: Compute best threshold (Youden's J)
opt_thresh <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")

# Create a named data.frame of metrics
metrics <- data.frame(
  Metric = c("AUC", "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score"),
  Value = round(c(roc_auc, accuracy, sensitivity, specificity, precision, f1_score), 4)
)

# ---- Save model and metrics ----
saveRDS(
  list(
    model = xgb_model,
    metrics = metrics,
    best_threshold = opt_thresh
  ),
  file = "xgb_model_with_metrics.rds"
)

# ---- Load model and metrics later ----
loaded <- readRDS("xgb_model_with_metrics.rds")
loaded_model <- loaded$model
loaded_metrics <- loaded$metrics
loaded_threshold <- loaded$best_threshold

# ---- Output: Print metrics ----
cat("📊 Model Performance Metrics:\n")
print(loaded_metrics)
# When calculating the threshold:
# Compute best threshold using Youden’s J (returns named vector)
opt_coords <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity", "precision", "recall", "accuracy"), best.method = "youden")
opt_thresh <- as.numeric(opt_coords["threshold"])  # Extract just the threshold
model_bundle <- list(
  model = xgb_model,
  metrics = metrics,
  best_threshold = opt_thresh,
  threshold_metrics = opt_coords,
  testData =testData,
)

saveRDS(model_bundle, file = "xgb_model_with_metrics.rds")
loaded <- readRDS("xgb_model_with_metrics.rds")

cat("📊 Model Performance Metrics:\n")
print(loaded$metrics)

cat("\n🔍 Best threshold (Youden's J):", round(loaded$best_threshold, 3), "\n")

cat("\n📐 Threshold-specific performance at", round(loaded$best_threshold, 3), ":\n")
print(round(loaded$threshold_metrics, 3))