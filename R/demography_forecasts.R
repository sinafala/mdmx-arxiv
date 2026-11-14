# ══════════════════════════════════════════════════════════════════════════════
# demography_forecasts.R — comparator forecasts with the R package `demography`
# ══════════════════════════════════════════════════════════════════════════════
#
# Used by book chapter 11-forecasting (paper §11.5).  For each job (a
# population and a forecast origin) three established methods are fitted to
# the population's HMD deaths and exposures up to its last observed year at
# or before the origin, and forecast to origin + horizon:
#
#   LC    Lee–Carter (Lee & Carter 1992): demography::lca on the raw death
#         rates with the k_t adjustment `lc_adjust`, k_t forecast by a random
#         walk with drift; each sex separately.  Zero rates are replaced by
#         lca's own interpolation (interpolate = TRUE).
#   HU    Hyndman–Ullah (2007): functional principal components of the
#         smoothed log rates (demography::fdm, order `fdm_order`), each
#         coefficient forecast by `fdm_ts_method` (auto ARIMA); each sex
#         separately.
#   CFDM  Coherent functional demographic model (Hyndman, Booth & Yasmeen
#         2013): demography::coherentfdm, the product–ratio method — fdm of
#         the geometric mean of the female and male rates (ARIMA) and of the
#         two sex ratios (stationary ARFIMA), so that the forecast sex ratios
#         do not diverge.
#
# The rates are deaths / exposures from the HMD (open age group at
# `open_age`); the smoothed rates come from hmd_smooth.R.  Years without an
# observed schedule inside a series were already filled (by linear
# interpolation of deaths and exposures) in Python, because these methods
# need an unbroken annual series.
#
# No error is ignored: every failed fit is written to failures.csv, and every
# warning R raises to warnings.csv; the chapter reports both.
#
# Called from Python through mdmx.rbridge:
#     Rscript --vanilla R/demography_forecasts.R <work folder>
#
# Input   rates.csv     pop, sex, year, age, deaths, exposure
#         smoothed.csv  pop, sex, year, age, mx
#         jobs.csv      pop, origin, first_year, last_year, h
#         settings.json the [forecast.comparators] parameters
# Output  forecasts.csv method, pop, origin, sex, year, a0 … a<open_age>  (m_x)
#         failures.csv  method, pop, origin, message
#         warnings.csv  method, pop, origin, message
#         sessionInfo.txt
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
    library(demography)
    library(data.table)
    library(jsonlite)
})

work <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(work)) stop("Usage: Rscript demography_forecasts.R <work folder>")
set.seed(1)    # the methods are deterministic; this only guards against surprises

cfg <- fromJSON(file.path(work, "settings.json"))
rates <- fread(file.path(work, "rates.csv"))
smooth <- fread(file.path(work, "smoothed.csv"))
jobs <- fread(file.path(work, "jobs.csv"))
ages <- sort(unique(rates$age))
sexes <- c("female", "male")
cat(sprintf("%d jobs; ages %d-%d\n", nrow(jobs), min(ages), max(ages)))

# Matrix (ages × years) of one column for one population, sex and span of years.
as_matrix <- function(tab, p, s, years, column) {
    x <- tab[pop == p & sex == s & year %in% years]
    m <- matrix(NA_real_, length(ages), length(years))
    m[cbind(match(x$age, ages), match(x$year, years))] <- x[[column]]
    if (anyNA(m) && column != "rate") stop(sprintf("missing %s for %s %s", column, p, s))
    m
}

# A two-sex demogdata object from rate and exposure matrices.
two_sex <- function(rate, pop, years, label) {
    dd <- demogdata(data = rate$female, pop = pop$female, ages = ages, years = years,
                    type = "mortality", label = label, name = "female")
    dd$rate$male <- rate$male
    dd$pop$male <- pop$male
    dd
}

# Forecast m_x (ages × h) as rows: method, pop, origin, sex, year, a0 … aN.
emit <- function(method, job, sex, mat) {
    years_fc <- job$last_year + seq_len(ncol(mat))
    m <- t(mat)
    colnames(m) <- paste0("a", ages)
    data.table(method = method, pop = job$pop, origin = job$origin, sex = sex, year = years_fc, m)
}

forecasts <- list(); failures <- list(); warns <- list()

# Run one method for one job; record (never swallow) errors and warnings.
run_method <- function(method, job, expr) {
    withCallingHandlers(
        tryCatch(expr, error = function(e) {
            failures[[length(failures) + 1]] <<- data.table(
                method = method, pop = job$pop, origin = job$origin, message = conditionMessage(e))
            NULL
        }),
        warning = function(w) {
            warns[[length(warns) + 1]] <<- data.table(
                method = method, pop = job$pop, origin = job$origin, message = conditionMessage(w))
            invokeRestart("muffleWarning")
        })
}

started <- proc.time()[["elapsed"]]
for (i in seq_len(nrow(jobs))) {
    job <- jobs[i]
    years <- job$first_year:job$last_year
    raw_rate <- raw_pop <- sm_rate <- list()
    for (s in sexes) {
        deaths <- as_matrix(rates, job$pop, s, years, "deaths")
        raw_pop[[s]] <- as_matrix(rates, job$pop, s, years, "exposure")
        raw_rate[[s]] <- ifelse(raw_pop[[s]] > 0, deaths / raw_pop[[s]], NA_real_)
        sm_rate[[s]] <- as_matrix(smooth[, .(pop, sex, year, age, rate = mx)], job$pop, s, years, "rate")
    }
    raw <- two_sex(raw_rate, raw_pop, years, job$pop)
    smoothed <- two_sex(sm_rate, raw_pop, years, job$pop)

    out <- run_method("LC", job, {
        lapply(sexes, function(s) {
            fit <- lca(raw, series = s, max.age = max(ages), adjust = cfg$lc_adjust,
                       interpolate = TRUE)
            fc <- forecast(fit, h = job$h, jumpchoice = cfg$lc_jumpchoice)
            emit("LC", job, s, fc$rate[[s]])
        })
    })
    if (!is.null(out)) forecasts <- c(forecasts, out)

    out <- run_method("HU", job, {
        lapply(sexes, function(s) {
            fit <- fdm(smoothed, series = s, order = cfg$fdm_order, method = cfg$fdm_method)
            fc <- forecast(fit, h = job$h, method = cfg$fdm_ts_method, jumpchoice = cfg$fdm_jumpchoice)
            emit("HU", job, s, fc$rate[[s]])
        })
    })
    if (!is.null(out)) forecasts <- c(forecasts, out)

    out <- run_method("CFDM", job, {
        fit <- coherentfdm(smoothed, order1 = cfg$coherent_order_product,
                           order2 = cfg$coherent_order_ratio)
        fc <- forecast(fit, h = job$h)
        lapply(sexes, function(s) emit("CFDM", job, s, fc[[s]]$rate[[s]]))
    })
    if (!is.null(out)) forecasts <- c(forecasts, out)

    if (i %% 10 == 0 || i == nrow(jobs)) {
        cat(sprintf("  [%d/%d] %.0f s\n", i, nrow(jobs), proc.time()[["elapsed"]] - started))
    }
}

fwrite(rbindlist(forecasts), file.path(work, "forecasts.csv"))
empty <- data.table(method = character(), pop = character(), origin = integer(), message = character())
fwrite(if (length(failures)) rbindlist(failures) else empty, file.path(work, "failures.csv"))
fwrite(if (length(warns)) rbindlist(warns) else empty, file.path(work, "warnings.csv"))
cat(sprintf("Done: %d failures, %d warnings\n", length(failures), length(warns)))
writeLines(capture.output(print(sessionInfo())), file.path(work, "sessionInfo.txt"))
