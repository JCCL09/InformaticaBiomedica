# run_api.R
library(plumber)

options_plumber(legacyRedirects = FALSE)

.get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", cmd_args[grep("^--file=", cmd_args)])
  if (length(file_arg) == 1) {
    return(dirname(normalizePath(file_arg)))
  }

  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile) && nzchar(ofile)) {
    return(dirname(normalizePath(ofile)))
  }

  getwd()
}

base_dir <- .get_script_dir()
setwd(base_dir)

pr <- plumb("plumber.R")
pr$run(host = "127.0.0.1", port = 1025, swagger = TRUE)
