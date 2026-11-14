# ══════════════════════════════════════════════════════════════════════════════
# mortcast_schedules.R — schedules for bayesLife's e0 forecasts (MortCast)
# ══════════════════════════════════════════════════════════════════════════════
#
# Used by book chapter 12-evaluation (decision 13, Sam, 2026-10-29/30).  bayesLife forecasts life
# expectancy only.  To score it on schedules (the age pattern and the age-specific sex differences),
# every population's median female and male e0 path is turned into death-rate schedules by
# MortCast's coherent rotated Lee–Carter method, the UN's own route (Ševčíková et al. 2016):
#
#   * lileecarter.estimate(): coherent Lee–Carter parameters (Li & Lee 2005) from the population's
#     own single-year death rates over the job's fitting window, female and male sharing b_x, with
#     a_x the log rates of the window's last year (ax.index = ncol(mx), an option MortCast's own
#     documentation shows; its default, the mean over the window, would start every forecast from a
#     shape that mixes decades, from 1751 for Sweden);
#   * mortcast(rotate = TRUE): for every forecast year, the level of mortality whose life
#     expectancy is the given e0, with b_x rotated towards its ultimate shape as e0 rises
#     (Li, Lee & Gerland 2013).
#
# The rates are the comparators' smoothed rates (each year smoothed on its own, so no look-ahead):
# a log-rate model cannot take the zero death counts of the raw rates.  MortCast's other arguments
# are its defaults.  A population that fails stops the run.
#
# Called from Python through mdmx.rbridge:
#     Rscript --vanilla R/mortcast_schedules.R <work folder>
#
# Input   rates.csv     pop, sex, year, age, mx: each population's rates over its fitting window
#                       (ages 0 … 100, the last the open group)
#         e0.csv        pop, year, female, male: the median e0 forecast
#         settings.json rotate
# Output  mx.csv        pop, sex, year, age, mx: the forecast rates (the ages of rates.csv)
#         failures.csv, warnings.csv, sessionInfo.txt
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
    library(MortCast)
    library(data.table)
    library(jsonlite)
})

work <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(work)) stop("Usage: Rscript mortcast_schedules.R <work folder>")
cfg <- fromJSON(file.path(work, "settings.json"))
warns <- list()

# Ages × years matrix of one population and sex, with the ages as row names (MortCast's convention).
rate_matrix <- function(r) {
    w <- dcast(r, age ~ year, value.var = "mx")
    m <- as.matrix(w[, -1, with = FALSE])
    rownames(m) <- w$age
    if (anyNA(m) || any(m <= 0)) stop("rates missing or not positive")
    m
}

result <- withCallingHandlers(tryCatch({
    rates <- fread(file.path(work, "rates.csv"))
    e0 <- fread(file.path(work, "e0.csv"))
    out <- list()
    for (p in unique(e0$pop)) {
        r <- rates[pop == p]
        if (!nrow(r)) stop("no rates for ", p)
        m_male <- rate_matrix(r[sex == "male"])
        m_female <- rate_matrix(r[sex == "female"])
        if (ncol(m_male) != ncol(m_female)) stop(p, ": the sexes' rates cover different years")
        # a_x from the window's last year (column ncol: the years are in calendar order).
        lc <- tryCatch(lileecarter.estimate(m_male, m_female, nx = 1, ax.index = ncol(m_male)),
                       error = function(e) stop(p, ": lileecarter.estimate failed: ", conditionMessage(e)))
        e <- e0[pop == p][order(year)]
        pred <- tryCatch(mortcast(e$male, e$female, lc, rotate = isTRUE(cfg$rotate)),
                         error = function(e) stop(p, ": mortcast failed: ", conditionMessage(e)))
        for (sex in c("female", "male")) {
            m <- pred[[sex]]$mx                                        # ages × forecast years
            if (ncol(m) != nrow(e)) stop(p, ": mortcast returned ", ncol(m), " years, not ", nrow(e))
            out[[length(out) + 1]] <- data.table(pop = p, sex = sex, year = rep(e$year, each = nrow(m)),
                                                 age = rep(lc$ages, times = ncol(m)), mx = as.vector(m))
        }
    }
    rbindlist(out)
}, error = function(e) {
    fwrite(data.table(message = conditionMessage(e)), file.path(work, "failures.csv"))
    fwrite(if (length(warns)) rbindlist(warns) else data.table(message = character(0)), file.path(work, "warnings.csv"))
    writeLines(capture.output(sessionInfo()), file.path(work, "sessionInfo.txt"))
    quit(status = 1)
}), warning = function(w) {
    warns[[length(warns) + 1]] <<- data.table(message = conditionMessage(w))
    invokeRestart("muffleWarning")
})

fwrite(result, file.path(work, "mx.csv"))
fwrite(data.table(message = character(0)), file.path(work, "failures.csv"))
fwrite(if (length(warns)) rbindlist(warns) else data.table(message = character(0)), file.path(work, "warnings.csv"))
writeLines(capture.output(sessionInfo()), file.path(work, "sessionInfo.txt"))
