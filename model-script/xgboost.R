if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, tidymodels, janitor,
               skimr, tictoc, vip)

dat <- read_csv("data/heart_2020_cleaned.csv") |>
  clean_names()

# skim(dat)

dat <- dat |>
  mutate(heart_disease = as_factor(heart_disease),
         smoking = as_factor(smoking),
         alcohol_drinking = as_factor(alcohol_drinking),
         stroke = as_factor(stroke),
         diff_walking = as_factor(diff_walking),
         sex = as_factor(sex),
         age_category = factor(age_category,
                               levels = c("18-24", "25-29", "30-34", "35-39",
                                          "40-44", "45-49", "50-54", "55-59",
                                          "60-64", "65-69", "70-74", "75-79",
                                          "80 or older")),
         race = as_factor(race),
         diabetic = as_factor(diabetic),
         physical_activity = as_factor(physical_activity),
         gen_health = factor(gen_health,
                                levels = c("Poor", "Fair", "Good", "Very good", "Excellent")),
         asthma = as_factor(asthma),
         kidney_disease = as_factor(kidney_disease),
         skin_cancer = as_factor(skin_cancer))


tidymodels_prefer()

set.seed(2022)
heart_split <- initial_split(data = dat, prop = 0.8)
heart_train <- training(heart_split)
heart_test <- testing(heart_split)
heart_metrics <- metric_set(accuracy, roc_auc, mn_log_loss)

set.seed(2023)
heart_folds <- vfold_cv(heart_train, v = 10)

heart_train |>
  ggplot(aes(x = bmi, y = physical_health, color = heart_disease)) +
  geom_point(alpha = 0.5, size = 0.5) +
  labs(x = "BMI", y = "Physical Health") +
  theme_minimal()

heart_train |>
  ggplot(aes(x = age_category, fill = heart_disease)) +
  geom_bar(alpha = 0.5, position = "identity") +
  labs(x = "Age Category", fill = NULL) +
  theme_minimal()

heart_recipe <- recipe(heart_disease ~ ., data = heart_train) |>
  step_dummy(all_nominal_predictors(), one_hot=TRUE) |>
  step_zv(all_predictors()) # precautionary step to remove variables only containing a single value

prep(heart_recipe)

# Tunable xgboost model with early stopping

stopping_spec <- boost_tree(
  trees = tune(), mtry = tune(), learn_rate = tune(), stop_iter = tune()
) |>
  set_engine("xgboost", validation = 0.1) |>
  set_mode("classification")

stopping_grid <- grid_latin_hypercube(
  trees(range = c(25, 200)),
  mtry(range = c(2L, 10L)),
  learn_rate(range = c(-5, -1)),
  stop_iter(range = c(10L, 50L)),
  size = 10
)

# Workflow

early_stop_wf <- workflow(heart_recipe, stopping_spec)

doParallel::registerDoParallel()

set.seed(1234)
tic("total")
tic("tune grid")
stopping_rs <- tune_grid(
  early_stop_wf,
  heart_folds,
  grid = stopping_grid,
  metrics = heart_metrics
)
toc()
toc()

# approximately 1000 seconds

autoplot(stopping_rs) +
  geom_line() +
  theme_minimal()

show_best(stopping_rs, metric = "roc_auc")

stopping_fit <- early_stop_wf |>
  finalize_workflow(select_best(stopping_rs, "roc_auc")) |>
  last_fit(heart_split)

collect_metrics(stopping_fit)

extract_workflow(stopping_fit) |>
  extract_fit_parsnip() |>
  vip(num_features = 17, geom = "point") +
  theme_minimal()

collect_predictions(stopping_fit) |>
  roc_curve(heart_disease, .pred_No) |>
  ggplot(aes(1 - specificity, sensitivity)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(alpha = 0.8, size = 1, color = "royalblue") +
  coord_equal() +
  labs(color = NULL)

collect_predictions(stopping_fit) |>
  conf_mat(heart_disease, .pred_class) |>
  autoplot()

save.image("data/xgboost_w_early_stopping.RData")
load("data/xgboost_w_early_stopping.RData")