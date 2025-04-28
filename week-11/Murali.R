library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                            1, 0)


library(xgboost)
library(caret)
library(Metrics)
library(tictoc)

# Function to train and evaluate XGBoost at different dataset sizes
evaluate_xgboost <- function(data, sizes = c(100, 1000, 10000, 100000, 1000000, 10000000)) {
  # Convert outcome to factor (if not already)
  data$outcome <- as.factor(data$outcome)
  results <- data.frame()
  
  for (size in sizes) {
    # Check if we have enough data for this size
    if (size > nrow(data)) {
      cat(paste("Warning: Requested size", size, "exceeds available data (", nrow(data), "). Skipping.\n"))
      next
    }
    
    # Sample the dataset to the specified size
    set.seed(123) # For reproducibility
    sample_indices <- sample(1:nrow(data), size = size)
    sample_data <- data[sample_indices, ]
    
    # Split into training (70%) and testing (30%) sets
    train_indices <- createDataPartition(sample_data$outcome, p = 0.7, list = FALSE)
    train_data <- sample_data[train_indices, ]
    test_data <- sample_data[-train_indices, ]
    
    # Prepare matrices for XGBoost
    predictors <- setdiff(names(sample_data), "outcome")
    dtrain <- xgb.DMatrix(data = as.matrix(train_data[, predictors]), 
                          label = as.numeric(as.character(train_data$outcome)))
    dtest <- xgb.DMatrix(data = as.matrix(test_data[, predictors]), 
                         label = as.numeric(as.character(test_data$outcome)))
    
    # Set XGBoost parameters
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      eta = 0.1,
      max_depth = 6,
      min_child_weight = 1,
      subsample = 0.8,
      colsample_bytree = 0.8
    )
    
    # Cross-validation setup
    nrounds <- 100
    nfold <- 5
    
    # Start timing
    tic(paste("XGBoost with size", size))
    
    # Perform cross-validation
    cv_model <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      nfold = nfold,
      early_stopping_rounds = 10,
      verbose = 0
    )
    
    # Train the final model with the optimal number of rounds
    best_nrounds <- cv_model$best_iteration
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = best_nrounds,
      watchlist = list(train = dtrain, test = dtest),
      verbose = 0
    )
    
    # Stop timing and get elapsed time
    time_taken <- toc(log = FALSE, quiet = TRUE)
    elapsed_time <- time_taken$toc - time_taken$tic
    
    # Make predictions on test set
    predictions_prob <- predict(model, dtest)
    predictions <- ifelse(predictions_prob > 0.5, 1, 0)
    actual <- as.numeric(as.character(test_data$outcome))
    
    # Calculate metrics
    accuracy <- mean(predictions == actual)
    auc_value <- auc(actual, predictions_prob)
    
    # Store results
    result <- data.frame(
      Method = "XGBoost with simple cross-validation",
      Dataset_size = size,
      Test_accuracy = round(accuracy, 4),
      Test_AUC = round(auc_value, 4),
      Time_taken_seconds = round(elapsed_time, 2),
      Best_iterations = best_nrounds
    )
    
    # Display results
    print(result)
    
    # Append to results dataframe
    results <- rbind(results, result)
  }
  
  return(results)
}

# Run the evaluation (assuming dfdata exists)
results <- evaluate_xgboost(dfdata)

# Format results for the table
formatted_results <- results[, c("Dataset_size", "Test_accuracy", "Test_AUC", "Time_taken_seconds", "Best_iterations")]
print(formatted_results)


evaluate_xgboost_caret <- function(data, sizes = c(100, 1000, 10000, 100000, 1000000, 10000000)) {
  # Create a copy of the data
  data_copy <- data
  
  # Convert outcome to factor with valid R variable names
  data_copy$outcome <- factor(data_copy$outcome, 
                              levels = c(0, 1), 
                              labels = c("neg", "pos"))
  
  results <- data.frame()
  
  # Define the predictors and outcome
  predictors <- setdiff(names(data_copy), "outcome")
  
  for (size in sizes) {
    # Check if we have enough data
    if (size > nrow(data_copy)) {
      cat(paste("Warning: Requested size", size, "exceeds available data. Skipping.\n"))
      next
    }
    
    # Sample the dataset
    set.seed(123)
    sample_indices <- sample(1:nrow(data_copy), size = size)
    sample_data <- data_copy[sample_indices, ]
    
    # Split into training (70%) and testing (30%)
    train_indices <- createDataPartition(sample_data$outcome, p = 0.7, list = FALSE)
    train_data <- sample_data[train_indices, ]
    test_data <- sample_data[-train_indices, ]
    
    # Set up 5-fold cross-validation
    ctrl <- trainControl(
      method = "cv",
      number = 5,
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      verboseIter = FALSE
    )
    
    # Set up simple XGBoost parameters
    xgb_params <- expand.grid(
      nrounds = 100,
      eta = 0.1,
      max_depth = 6,
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    )
    
    # Start timing
    tic(paste("XGBoost caret with size", size))
    
    # Train model
    model <- train(
      x = train_data[, predictors],
      y = train_data$outcome,
      method = "xgbTree",
      trControl = ctrl,
      tuneGrid = xgb_params,
      metric = "ROC"
    )
    
    # Timing end
    time_taken <- toc(log = FALSE, quiet = TRUE)
    elapsed_time <- time_taken$toc - time_taken$tic
    
    # Make predictions
    predictions_prob <- predict(model, test_data[, predictors], type = "prob")[, "pos"]
    predictions <- predict(model, test_data[, predictors])
    actual <- test_data$outcome
    
    # Calculate metrics
    accuracy <- mean(predictions == actual)
    auc_value <- auc(ifelse(actual == "pos", 1, 0), predictions_prob)
    
    # Store results
    result <- data.frame(
      Method = "XGBoost via caret with 5-fold CV",
      Dataset_size = size,
      Test_accuracy = round(accuracy, 4),
      Test_AUC = round(auc_value, 4),
      Time_taken_seconds = round(elapsed_time, 2)
    )
    
    # Display results
    print(result)
    
    # Append to results dataframe
    results <- rbind(results, result)
  }
  
  return(results)
}

# Run the evaluation
results <- evaluate_xgboost_caret(dfdata)

# Format results as a table
formatted_table <- data.frame(
  "Dataset_size" = results$Dataset_size,
  "Test_accuracy" = results$Test_accuracy,
  "Test_AUC" = results$Test_AUC,
  "Time_taken_seconds" = results$Time_taken_seconds
)

print(formatted_table)



