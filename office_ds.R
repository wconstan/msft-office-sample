# install.packages(c('data.table', 'randomForest', 'caret', 'ggplot2', 'e1071','DMwR','ROSE'))

###
# Load packages
library(data.table)
library(randomForest)
library(caret)
library(ggplot2)
library(e1071)
library(DMwR)
library(ROSE)

###
# Define local functions
make_pairs <- function(data){
    grid <- expand.grid(x = 1:ncol(data), y = 1:ncol(data))
    grid <- subset(grid, x != y)
    all <- do.call("rbind", lapply(1:nrow(grid), function(i) {
        xcol <- grid[i, "x"]
        ycol <- grid[i, "y"]
        data.frame(xvar = names(data)[ycol], yvar = names(data)[xcol], 
                   x = data[, xcol], y = data[, ycol], data)
    }))
    all$xvar <- factor(all$xvar, levels = names(data))
    all$yvar <- factor(all$yvar, levels = names(data))
    densities <- do.call("rbind", lapply(1:ncol(data), function(i) {
        data.frame(xvar = names(data)[i], yvar = names(data)[i], x = data[, i])
    }))
    list(all=all, densities=densities)
}

pairs_plot <- function(data, dep_var='Outcome', standardize = TRUE){
    
    if (standardize){
        for (j in setdiff(names(data), dep_var)){ 
            set(data, j = j, value = scale(data[[j]]))}
    }
    
    ind_vars = setdiff(names(data), dep_var)
    df = data.frame(data)
    gg = make_pairs(df[ind_vars])
    plot_data = gg$all
    plot_data[[dep_var]] = rep(df[[dep_var]], length=nrow(gg$all))
    
    # pairs plot
    g <- ggplot(plot_data, aes_string(x = "x", y = "y")) + 
        facet_grid(xvar ~ yvar, scales = "free") + 
        geom_point(aes(colour=plot_data[[dep_var]]), na.rm = TRUE, alpha=0.8, size=0.5) + 
        stat_density(aes(x = x, y = ..scaled.. * diff(range(x)) + min(x)), 
                     data = gg$densities, position = "identity", 
                     colour = "grey20", geom = "line") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
        labs(x='', y='', color=dep_var)

    list(plot=g, data=data)
}

majority_class_error <- function(...){
    # This function assumes the inputs are data.table
    # objects, each with a column named 'Outcome'. It calculates
    # the classification error by first forming a frequency
    # table over Outcome, finding the (label of) the majority class,
    # and forming a sum of the number of observations that are
    # NOT in the majority class over all data.table inputs, resulting
    # in the total number of errors (num_err). Normalizing the number
    # of errors by the total number of observations (num_obs)
    # results in the classification error for the inputs.
    num_err = num_obs = 0
        
    for (data in list(...)){
        tbl = data[, .N, Outcome]
        majority_class = as.character(tbl[which.max(N), Outcome])
        num_err = num_err + max(tbl[Outcome != majority_class, N], 0)
        num_obs = num_obs + tbl[, sum(N)]
    }
    
    return(num_err/num_obs)
}

score_model <- function(model, new_data, dep_var='Outcome'){
    actual_df <- data.frame(actual = new_data[[dep_var]])
    score_df <- predict(model, newdata = new_data, type = "prob")
    levels <- names(score_df)
    names(score_df) <- paste0('P(', dep_var, '=', levels, ')')
    pred_df <- data.frame(pred = apply(score_df, MARGIN = 1, function(x){
        factor(levels[which.max(x)], level=levels)}))
    return(data.table(cbind(actual_df, score_df, pred_df)))
}

confusion_matrix <- function(score_df){
    confusionMatrix(score_df$pred, score_df$actual)
}

fit_model <- function(form,
                      classifier = "rf",
                      training_data, 
                      validation_data, 
                      sampling="none",
                      method = "repeatedcv", # repeated cross-valiation
                      repeats = 5, # 5 repeats of ...
                      folds = 10, # 10-fold validation
                      pre_process = c("scale", "center"), # scale and recenter data, a priori
                      verbose = FALSE,
                      seed = NULL, 
                      ...){
    
    sampling <- match.arg(tolower(sampling),
                          choices = c("none", "down", "up", "smote", "rose"))
    
    ctrl <- trainControl(method = method, 
                         sampling = if (sampling == "none") NULL else sampling,
                         number = folds, 
                         repeats = repeats, 
                         verboseIter = verbose)
    
    if (!is.null(seed)) set.seed(seed)
    
    model <- caret::train(form,
                          data = training_data,
                          method = classifier,
                          preProcess = pre_process,
                          trControl = ctrl, 
                          ...)

    score <- score_model(model, validation_data)
    confusion <- confusion_matrix(score)
    list(model=model, score=score, confusion=confusion)
}

add_noise <- function(x, snr=2){
    noise <- rnorm(x)
    k <- sqrt(var(x)/(snr*var(noise)))
    x + k * noise 
}

get_metrics <- function(obj,  
                        metrics = c('Sensitivity','Specificity','Precision',
                                    'Recall','F1','Balanced Accuracy'),
                        snr = NA,
                        classifier = 'rf'){
    obj_name <- deparse(substitute(obj))
    sampling_methods <- names(obj)
    
    metric_dt <- do.call('cbind', lapply(sampling_methods, function(s) {
        data = data.table(obj[[s]]$confusion$byClass[metrics])
        setnames(data, s)
        return(data)
    }))
    
    snr_levels = paste0('snr = ', c(NA, 2, 1))
    
    metrics[length(metrics)] <- "Accuracy"
    metric_dt[, classifier := classifier]
    metric_dt[, metric := factor(metrics, levels = metrics)]
    metric_dt[, snr := factor(sprintf('snr = %s', snr), levels = snr_levels)]
    melt(data=metric_dt, 
         id.vars = c('classifier', 'metric', 'snr'), 
         variable.name = 'sampling')
}

###
# load data
dt <- data.table::fread('test_dataset.csv')
dt$Outcome = dt[, factor(Outcome)]

###
# Define regression formula
dep_var = 'Outcome'
form = formula(paste(dep_var, '~ .'))
print(form)

###
# impute missing data
if (any(is.na(dt))){
    dt <- data.table(rfImpute(form, dt))
    for (j in setdiff(names(dt), dep_var)){ 
        set(dt, j = j, value = as.integer(dt[[j]]))}
}

###
# Generate a pairs plot
# Note: standardize data for better visual comparison
scatter <- pairs_plot(dt, dep_var = dep_var, standardize = TRUE)
print(scatter$plot)
dt_standardized = scatter$data

# By inspection of the pairs plot, V9 does a great job in dividing the
# Outcomes across all explanatory variables. For example, here is the division
# for (scaled) V9 vs V1
v9_cut = dt_standardized[Outcome == 0, min(V9)]
dt_standardized[, plot(V1, V9, type='p', col=Outcome, pch=19)]
abline(h=v9_cut)
title(sprintf('split on standardized data: V9 = %0.3f', v9_cut))

# Assess the classification error given a cut of the
# data at v9_cut and compare it to original data using
# majority class as predictor. In fact, cutting V9 at 
# just the right value will result in obtaining 
# a zero classification error using all of the original data
majority_class_error_original = majority_class_error(dt_standardized)
majority_class_error_v9split = majority_class_error(dt_standardized[V9 >= v9_cut], 
                                                    dt_standardized[V9 < v9_cut])
sprintf('classification error: original = %.2f, (V9 >= %.3f) = %.2f', 
        majority_class_error_original,
        v9_cut,
        majority_class_error_v9split)

###
# create training and validation data sets using a 70/30 split
set.seed(10) # set seed for replicability 
index <- createDataPartition(dt[[dep_var]], p = 0.7, list = FALSE)
training_data <- dt[index, ]
validation_data  <- dt[-index, ]

###
# Fit various random forest models, using caret's sampling functions
# to address class imbalance.
# See https://topepo.github.io/caret/subsampling-for-class-imbalances.html for details

sampling_methods = c("none","up","down","rose","smote")

rf = sapply(sampling_methods, function(sampling){
    fit_model(form=form,
              classifier = "rf",
              training_data = training_data, 
              validation_data = validation_data,
              sampling = sampling, 
              seed = 500)
}, simplify = FALSE)

logit = sapply(sampling_methods, function(sampling){
    fit_model(form=form,
              classifier = "glm", family="binomial",
              training_data = training_data, 
              validation_data = validation_data,
              sampling = sampling, 
              seed = 100)
}, simplify = FALSE)

###
# Add Gaussian white noise to V9 to test the efficacy of our models to
# noisy V9 data, which currently does not exist in our training data
training_data_snr_2 <- copy(training_data)
training_data_snr_2$V9 = add_noise(training_data_snr_2$V9, snr=2)

scatter_snr_2 <- pairs_plot(training_data_snr_2, dep_var = dep_var, standardize = TRUE)
print(scatter_snr_2$plot)

###
# Re-fit models using noisy training data
rf_snr_2 = sapply(sampling_methods, function(sampling){
    fit_model(form=form,
              classifier = "rf",
              training_data = training_data_snr_2, 
              validation_data = validation_data,
              sampling = sampling, 
              seed = 5)
}, simplify = FALSE)

logit_snr_2 = sapply(sampling_methods, function(sampling){
    fit_model(form=form,
              classifier = "glm", family="binomial",
              training_data = training_data_snr_2, 
              validation_data = validation_data,
              sampling = sampling, 
              seed = 10)
}, simplify = FALSE)

###
# SNR = 1
training_data_snr_1 <- copy(training_data)
training_data_snr_1$V9 = add_noise(training_data_snr_1$V9, snr=1)

scatter_snr_1 <- pairs_plot(training_data_snr_1, dep_var = dep_var, standardize = TRUE)
print(scatter_snr_1$plot)

rf_snr_1 = sapply(sampling_methods, function(sampling){
    fit_model(form=form,
              classifier = "rf",
              training_data = training_data_snr_1, 
              validation_data = validation_data,
              sampling = sampling, 
              seed = 5)
}, simplify = FALSE)

logit_snr_1 = sapply(sampling_methods, function(sampling){
    fit_model(form=form,
              classifier = "glm", family="binomial",
              training_data = training_data_snr_1, 
              validation_data = validation_data,
              sampling = sampling, 
              seed = 10)
}, simplify = FALSE)

###
# Collect performance metrics
metrics_dt <- rbindlist(list(
    get_metrics(rf, classifier = 'rf'),
    get_metrics(logit, classifier = 'logit'),
    get_metrics(rf_snr_2, classifier = 'rf', snr=2),
    get_metrics(logit_snr_2, classifier = 'logit', snr = 2),
    get_metrics(rf_snr_1, classifier = 'rf', snr = 1),
    get_metrics(logit_snr_1, classifier = 'logit', snr = 1)
    ))

ggplot(metrics_dt, aes(metric, value, colour = classifier)) +
    geom_point() + facet_grid(snr ~ sampling) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    labs(y='')

