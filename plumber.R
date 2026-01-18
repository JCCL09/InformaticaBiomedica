# plumber.R
library(plumber)
library(jsonlite)

# Carga bundle al iniciar API
BUNDLE_PATH <- file.path("artifacts", "model_bundle.rds") # nolint:

if (!file.exists(BUNDLE_PATH)) {
  stop(
    "No existe artifacts/model_bundle.rds. Primero ejecuta: Rscript train_models.R" # nolint:
  )
}

bundle <- readRDS(BUNDLE_PATH)

predictors <- bundle$predictors
thr_diabetes <- bundle$threshold_diabetes
thr_pred_cond <- bundle$threshold_prediabetes_cond
mA <- bundle$model_diabetes # nolint:
mB <- bundle$model_prediabetes_cond # nolint:

# Helpers
as_numeric_strict <- function(x) {
  v <- suppressWarnings(as.numeric(x))
  if (length(v) != 1 || is.na(v)) return(NA_real_)
  v
}

build_newdata <- function(body, predictors) {
  # body: list (JSON parse)
  # Retorna data.frame con 1 fila y columnas predictors, o error
  missing <- setdiff(predictors, names(body))
  if (length(missing) > 0) {
    return(list(
      ok = FALSE,
      error = paste0("Faltan variables: ", paste(missing, collapse = ", "))
    ))
  }

  row <- lapply(predictors, function(p) as_numeric_strict(body[[p]]))
  names(row) <- predictors

  if (any(is.na(unlist(row)))) {
    bad <- predictors[is.na(unlist(row))]
    return(list(
      ok = FALSE,
      error = paste0(
        "Variables no numéricas o NA tras conversión: ",
        paste(bad, collapse = ", ")
      )
    ))
  }

  list(ok = TRUE, newdata = as.data.frame(row))
}

#* @apiTitle Diabetes Risk API (Fase 1)
#* @apiDescription Predice diabetes y prediabetes con dos modelos GLM (enfoque jerárquico). Devuelve probabilidades y clases por umbral Youden.# nolint:

#* Health check
#* @get /health
#* @serializer json
function() {
  list(
    status = "ok",
    model_created_at = bundle$created_at,
    thresholds = list(
      diabetes = thr_diabetes,
      prediabetes_cond = thr_pred_cond
    )
  )
}

#* Esquema de entrada (features requeridas)
#* @get /schema
#* @serializer json
function() {
  list(
    required_fields = predictors,
    notes = list(
      input_type = "Todas las variables deben enviarse como numéricas.",
      prediabetes_probability = "p_prediabetes = (1 - p_diabetes) * p_prediabetes_cond" # nolint:
    )
  )
}

#* Predicción (diabetes / prediabetes / healthy)
#* @post /predict
#* @parser json
#* @serializer json
#* @param payload:string JSON como string (solo para Swagger UI). Si Swagger no permite body, pega aquí un JSON compacto.
function(req, res, payload = "") {
  body <- req$body

  # Swagger a veces no envía body; usa el parámetro payload (string) como fallback
  if ((is.null(body) || length(body) == 0) && nzchar(payload)) {
    body <- tryCatch(jsonlite::fromJSON(payload), error = function(e) NULL)
  }

  # Algunos clientes/swagger pueden entregar data.frame
  if (is.data.frame(body)) {
    body <- as.list(body[1, , drop = TRUE])
  }

  # Si llega como string JSON, intenta parsearlo
  if (is.character(body) && length(body) == 1 && nzchar(body)) {
    body <- tryCatch(jsonlite::fromJSON(body), error = function(e) NULL)
  }

  if (is.null(body) || length(body) == 0) {
    res$status <- 400
    return(list(error = "JSON body vacío o inválido. En Swagger, usa el parámetro 'payload' y pega un JSON compacto con todas las variables del /schema."))
  }

  built <- build_newdata(body, predictors)
  if (!built$ok) {
    res$status <- 400
    return(list(error = built$error))
  }

  newdata <- built$newdata

  # Modelo A: diabetes
  p_diabetes <- as.numeric(predict(mA, newdata = newdata, type = "response"))

  # Modelo B: prediabetes condicional
  p_pred_cond <- as.numeric(predict(mB, newdata = newdata, type = "response"))

  # Probabilidades coherentes
  p_prediabetes <- (1 - p_diabetes) * p_pred_cond
  p_healthy <- 1 - p_diabetes - p_prediabetes

  # Clases por umbral (Youden)
  class_diabetes <- as.integer(p_diabetes >= thr_diabetes)
  class_pred_cond <- as.integer(p_pred_cond >= thr_pred_cond)

  # Clase final jerárquica
  final_class <- if (class_diabetes == 1) {
    "diabetes"
  } else if (class_pred_cond == 1) {
    "prediabetes"
  } else {
    "healthy"
  }

  list(
    probabilities = list(
      p_diabetes = p_diabetes,
      p_prediabetes = p_prediabetes,
      p_healthy = p_healthy,
      p_prediabetes_cond = p_pred_cond
    ),
    classes = list(
      diabetes = class_diabetes,
      prediabetes_cond = class_pred_cond,
      final_class = final_class
    ),
    thresholds = list(
      diabetes = thr_diabetes,
      prediabetes_cond = thr_pred_cond
    )
  )
}
