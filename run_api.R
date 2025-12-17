# run_api.R
library(plumber)

pr <- plumber::plumb("plumber.R")

plumber::pr_run(
  pr,
  host = "127.0.0.1",
  port = 8000,
  docs = TRUE,
  quiet = FALSE
)
