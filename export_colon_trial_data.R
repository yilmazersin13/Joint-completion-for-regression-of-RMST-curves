#!/usr/bin/env Rscript

# Export the documented survival::colon trial data without changing any fields.
# The Python analysis performs the treatment selection and endpoint construction.

suppressPackageStartupMessages(library(survival))

colon_data <- get("colon", envir = asNamespace("survival"))
write.csv(colon_data, "colon_trial_raw.csv", row.names = FALSE)

cat("survival package:", as.character(packageVersion("survival")), "\n")
cat("rows:", nrow(colon_data), "\n")
cat("patients:", length(unique(colon_data$id)), "\n")
cat("records by endpoint and treatment:\n")
print(with(colon_data, table(etype, rx, useNA = "ifany")))
cat("missing values by field:\n")
print(colSums(is.na(colon_data)))
