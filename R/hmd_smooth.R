# ══════════════════════════════════════════════════════════════════════════════
# hmd_smooth.R — smooth HMD death rates, one year at a time
# ══════════════════════════════════════════════════════════════════════════════
#
# Used by book chapter 11-forecasting (paper §11.5) for the Hyndman–Ullah and
# coherent functional-data comparators, which model *smoothed* log death
# rates (Hyndman & Ullah 2007; Hyndman, Booth & Yasmeen 2013).
#
# demography::smooth.demogdata smooths each calendar year's age curve on its
# own (weighted regression splines, constrained to be monotonic above age
# 65, with weights proportional to the expected deaths).  A year's smoothed
# curve therefore uses no other year's data, so the curves are computed once
# here and reused at every forecast origin.  (The function also estimates an
# "observational variance" across years; it affects only interval widths and
# is not written out.)
#
# Called from Python through mdmx.rbridge:
#     Rscript --vanilla R/hmd_smooth.R <work folder>
#
# The smoother weights each cell by its deaths on the log scale, so a cell with no death drops out.
# The deaths it is given are therefore the pooled ones of chapter d5 where pooling changed the rate
# (mdmx.forecast.pooled_deaths: the pooled rate times the year's exposure); otherwise, in the smallest
# populations, the curve would be fitted through the positive cells only and lie above the truth.
#
# Input   rates.csv       pop, sex, year, age, deaths, exposure
#                         (ages 0..open_age; the last is the open group)
# Output  smoothed.csv    pop, sex, year, age, mx   (smoothed death rates)
#         failures.csv    pop, sex, message         (series that failed)
#         sessionInfo.txt
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
    library(demography)
    library(data.table)
    library(jsonlite)
})

work <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(work)) stop("Usage: Rscript hmd_smooth.R <work folder>")

d <- fread(file.path(work, "rates.csv"))
ages <- sort(unique(d$age))
cat(sprintf("Smoothing %d population-sex series, ages %d-%d\n",
            uniqueN(d[, .(pop, sex)]), min(ages), max(ages)))

# Matrix (ages × years) of one column of a population-sex series.
as_matrix <- function(s, column, years) {
    m <- matrix(NA_real_, length(ages), length(years))
    m[cbind(match(s$age, ages), match(s$year, years))] <- s[[column]]
    m
}

smoothed <- list()
failures <- list()
started <- proc.time()[["elapsed"]]
for (key in split(unique(d[, .(pop, sex)]), seq_len(uniqueN(d[, .(pop, sex)])))) {
    s <- d[pop == key$pop & sex == key$sex]
    years <- sort(unique(s$year))
    deaths <- as_matrix(s, "deaths", years)
    exposure <- as_matrix(s, "exposure", years)
    # A cell with no exposure has no rate (NA); smooth.demogdata gives NA
    # and zero rates weight zero (hence the pooled deaths of the input).
    rate <- ifelse(exposure > 0, deaths / exposure, NA_real_)
    result <- tryCatch({
        dd <- demogdata(data = rate, pop = exposure, ages = ages, years = years,
                        type = "mortality", label = key$pop, name = key$sex)
        sm <- smooth.demogdata(dd)
        data.table(pop = key$pop, sex = key$sex,
                   year = rep(years, each = length(ages)),
                   age = rep(ages, times = length(years)),
                   mx = as.vector(sm$rate[[1]]))
    }, error = function(e) {
        failures[[length(failures) + 1]] <<- data.table(pop = key$pop, sex = key$sex,
                                                        message = conditionMessage(e))
        NULL
    })
    if (!is.null(result)) smoothed[[length(smoothed) + 1]] <- result
}
cat(sprintf("Done in %.0f s; %d series failed\n",
            proc.time()[["elapsed"]] - started, length(failures)))

fwrite(rbindlist(smoothed), file.path(work, "smoothed.csv"))
fail <- if (length(failures)) rbindlist(failures) else
    data.table(pop = character(), sex = character(), message = character())
fwrite(fail, file.path(work, "failures.csv"))
writeLines(capture.output(print(sessionInfo())), file.path(work, "sessionInfo.txt"))
