# train_models.R
# Entrena GLM A y B, fija umbrales por Youden en VALIDACIÓN,
# evalúa en VALIDACIÓN y TEST, y guarda bundle con splits+métricas.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(pROC)
})

set.seed(20250101)

# ============
# Configuración
# ============
csv_path <- "data/diabetes_012_health_indicators_BRFSS2015.csv"  # Ajusta ruta
out_bundle_path <- "artifacts/model_bundle.rds"

p_train <- 0.70
p_val   <- 0.15
p_test  <- 0.15

if (abs((p_train + p_val + p_test) - 1) > 1e-9) {
  stop("Los porcentajes de split deben sumar 1.")
}

# =========================
# Utilidades (estratificado)
# =========================
stratified_split_ids <- function(y, p_train, p_val, seed = 1) {
  set.seed(seed)
  if (!all(y %in% c(0, 1))) stop("y debe ser binario 0/1.")

  idx_pos <- which(y == 1)
  idx_neg <- which(y == 0)

  sample_pos <- sample(idx_pos)
  sample_neg <- sample(idx_neg)

  n_pos <- length(sample_pos)
  n_neg <- length(sample_neg)

  n_pos_train <- floor(n_pos * p_train)
  n_pos_val   <- floor(n_pos * p_val)
  n_pos_test  <- n_pos - n_pos_train - n_pos_val

  n_neg_train <- floor(n_neg * p_train)
  n_neg_val   <- floor(n_neg * p_val)
  n_neg_test  <- n_neg - n_neg_train - n_neg_val

  idx_train <- c(
    sample_pos[1:n_pos_train],
    sample_neg[1:n_neg_train]
  )

  idx_val <- c(
    sample_pos[(n_pos_train + 1):(n_pos_train + n_pos_val)],
    sample_neg[(n_neg_train + 1):(n_neg_train + n_neg_val)]
  )

  idx_test <- c(
    sample_pos[(n_pos_train + n_pos_val + 1):(n_pos_train + n_pos_val + n_pos_test)],
    sample_neg[(n_neg_train + n_neg_val + 1):(n_neg_train + n_neg_val + n_neg_test)]
  )

  list(train = idx_train, val = idx_val, test = idx_test)
}

youden_threshold <- function(y_true, prob) {
  roc_obj <- pROC::roc(
    response = factor(y_true, levels = c(0, 1)),
    predictor = prob,
    quiet = TRUE,
    direction = "<"
  )

  thr <- as.numeric(pROC::coords(
    roc_obj,
    x = "best",
    best.method = "youden",
    ret = "threshold",
    transpose = FALSE
  ))

  list(roc = roc_obj, threshold = thr)
}

metrics_at_threshold <- function(y_true, prob, thr) {
  y_pred <- ifelse(prob >= thr, 1, 0)

  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)

  sensitivity <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  specificity <- if ((tn + fp) == 0) NA_real_ else tn / (tn + fp)
  accuracy    <- (tp + tn) / (tp + tn + fp + fn)
  precision   <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  f1          <- if (is.na(precision) || is.na(sensitivity) || (precision + sensitivity) == 0) NA_real_ else {
    2 * precision * sensitivity / (precision + sensitivity)
  }

  list(
    threshold = thr,
    tp = tp, tn = tn, fp = fp, fn = fn,
    sensitivity = sensitivity,
    specificity = specificity,
    accuracy = accuracy,
    precision = precision,
    f1 = f1
  )
}

evaluate_split <- function(model, df_split, thr) {
  prob <- as.numeric(predict(model, newdata = df_split, type = "response"))
  roc_obj <- pROC::roc(factor(df_split$y, levels = c(0, 1)), prob, quiet = TRUE, direction = "<")
  auc_val <- as.numeric(pROC::auc(roc_obj))
  met <- metrics_at_threshold(df_split$y, prob, thr)
  list(auc = auc_val, metrics = met)
}

# =================
# Carga de los datos
# =================
df <- readr::read_csv(csv_path, show_col_types = FALSE)

if (!("Diabetes_012" %in% names(df))) {
  stop("No encuentro la columna 'Diabetes_012'. Revisa que estés usando el CSV 0/1/2.")
}
if (!all(df$Diabetes_012 %in% c(0, 1, 2))) {
  stop("Diabetes_012 debe contener solo 0,1,2. Hay valores fuera de ese conjunto.")
}

# ID estable para reproducibilidad del split
df <- df %>% mutate(row_id = dplyr::row_number())

predictors <- setdiff(names(df), c("Diabetes_012", "row_id"))

# Tu decisión: ordinales como numéricas
df <- df %>%
  mutate(across(all_of(predictors), ~ as.numeric(.)))

# ==========================
# Modelo A: diabetes (2 vs 0/1)
# ==========================
df_A <- df %>%
  mutate(y = ifelse(Diabetes_012 == 2, 1, 0)) %>%
  select(row_id, y, all_of(predictors))

ids_A <- stratified_split_ids(df_A$y, p_train, p_val, seed = 20250101)

A_train <- df_A[ids_A$train, , drop = FALSE] %>% select(-row_id)
A_val   <- df_A[ids_A$val, , drop = FALSE] %>% select(-row_id)
A_test  <- df_A[ids_A$test, , drop = FALSE] %>% select(-row_id)

mA <- glm(y ~ ., data = A_train, family = binomial())

prob_A_val <- as.numeric(predict(mA, newdata = A_val, type = "response"))
you_A <- youden_threshold(A_val$y, prob_A_val)
thr_A <- you_A$threshold

eval_A_val  <- evaluate_split(mA, A_val, thr_A)
eval_A_test <- evaluate_split(mA, A_test, thr_A)

# ==============================
# Modelo B: prediabetes (1 vs 0) en subset {0,1}
# ==============================
df_B <- df %>%
  filter(Diabetes_012 %in% c(0, 1)) %>%
  mutate(y = ifelse(Diabetes_012 == 1, 1, 0)) %>%
  select(row_id, y, all_of(predictors))

ids_B <- stratified_split_ids(df_B$y, p_train, p_val, seed = 20250102)

B_train <- df_B[ids_B$train, , drop = FALSE] %>% select(-row_id)
B_val   <- df_B[ids_B$val, , drop = FALSE] %>% select(-row_id)
B_test  <- df_B[ids_B$test, , drop = FALSE] %>% select(-row_id)

mB <- glm(y ~ ., data = B_train, family = binomial())

prob_B_val <- as.numeric(predict(mB, newdata = B_val, type = "response"))
you_B <- youden_threshold(B_val$y, prob_B_val)
thr_B <- you_B$threshold

eval_B_val  <- evaluate_split(mB, B_val, thr_B)
eval_B_test <- evaluate_split(mB, B_test, thr_B)

# ====================
# Resumen en consola
# ====================
cat("\n=== MODELO A (Diabetes 2 vs {0,1}) ===\n")
cat("Threshold Youden (val):", thr_A, "\n")
cat("AUC (val): ", round(eval_A_val$auc, 4), "\n")
print(eval_A_val$metrics)
cat("AUC (test): ", round(eval_A_test$auc, 4), "\n")
print(eval_A_test$metrics)

cat("\n=== MODELO B (Prediabetes 1 vs 0 | subset {0,1}) ===\n")
cat("Threshold Youden (val):", thr_B, "\n")
cat("AUC (val): ", round(eval_B_val$auc, 4), "\n")
print(eval_B_val$metrics)
cat("AUC (test): ", round(eval_B_test$auc, 4), "\n")
print(eval_B_test$metrics)

# ====================
# Guardado del bundle
# ====================
dir.create("artifacts", showWarnings = FALSE, recursive = TRUE)

bundle <- list(
  created_at = as.character(Sys.time()),
  predictors = predictors,

  threshold_diabetes = thr_A,
  threshold_prediabetes_cond = thr_B,

  model_diabetes = mA,
  model_prediabetes_cond = mB,

  splits = list(
    A = list(
      train_row_id = df_A$row_id[ids_A$train],
      val_row_id   = df_A$row_id[ids_A$val],
      test_row_id  = df_A$row_id[ids_A$test]
    ),
    B = list(
      train_row_id = df_B$row_id[ids_B$train],
      val_row_id   = df_B$row_id[ids_B$val],
      test_row_id  = df_B$row_id[ids_B$test]
    )
  ),

  performance = list(
    A = list(val = eval_A_val, test = eval_A_test),
    B = list(val = eval_B_val, test = eval_B_test)
  )
)

saveRDS(bundle, out_bundle_path)
cat("\nBundle guardado en: ", out_bundle_path, "\n", sep = "")
