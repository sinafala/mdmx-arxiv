# ══════════════════════════════════════════════════════════════════════════════
# demography_intervals.R — interval forecasts of e0 by the `demography` comparators
# ══════════════════════════════════════════════════════════════════════════════
#
# Used by book chapter 11-forecasting for the overall comparison of the methods
# (task T5b; docs/changes/11-forecasting.md §9).  For each job it refits
# Lee–Carter, Hyndman–Ullah and the coherent FDM exactly as
# demography_forecasts.R does (same inputs, settings and calls; that script,
# which gives the chapter's point forecasts, is not changed) and returns each
# method's own predictive distribution of e0:
#
#   LC    the k_t forecast of forecast.lca (random walk with drift; standard
#         error with drift uncertainty, se = "innovdrift") and the age pattern
#         (log jump-off rates, b_x): every age moves with k_t, so Python maps
#         quantiles of k_t to e0 (mdmx.compare.lc_quantiles).
#   HU    `nsim` draws of demography::simulate() of the forecast (coefficient
#         paths from their ARIMA models plus resampled residual curves; package
#         defaults), each sex separately.
#   CFDM  `nsim` draws of simulate() of the coherent forecast (product and
#         ratios; the two sexes share the product draw, so they are joint).
#
# simulate() ignores its `seed` argument, so the seed is set with set.seed()
# before each call, from the job's own seed.  e0 of every draw is computed with
# the chapter's life-table convention (port of mdmx.compare.e0_from_mx: the
# open-age rate repeated to n_ages − 1, q = m/(1 + (1 − a)m), the logit floor,
# L0 = a0·l0 + (1 − a0)·l1, trapezoid L); the e0 of the refitted point forecast
# is written as well, so that the chapter can check it against its cached point
# forecasts.
#
# No error is ignored: every failed fit is written to failures.csv, and every
# warning to warnings.csv.
#
# Called from Python through mdmx.rbridge (mdmx.compare.comparator_intervals):
#     Rscript --vanilla R/demography_intervals.R <work folder>
#
# Input   rates.csv     pop, sex, year, age, deaths, exposure
#         smoothed.csv  pop, sex, year, age, mx
#         jobs.csv      pop, origin, first_year, last_year, h, seed_hu, seed_cfdm
#         settings.json the [forecast.comparators] parameters, nsim, levels, n_ages, ax, a0, q_min
# Output  quantiles.csv method, pop, origin, sex, year, level, e0   (HU, CFDM; sex female, male, both)
#         points.csv    method, pop, origin, sex, year, e0_point, n_finite
#         lc_kt.csv     pop, origin, sex, year, kt_mean, kt_se
#         lc_age.csv    pop, origin, sex, age, log_jump, bx
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
if (is.na(work)) stop("Usage: Rscript demography_intervals.R <work folder>")

cfg <- fromJSON(file.path(work, "settings.json"))
rates <- fread(file.path(work, "rates.csv"))
smooth <- fread(file.path(work, "smoothed.csv"))
jobs <- fread(file.path(work, "jobs.csv"))
ages <- sort(unique(rates$age))
sexes <- c("female", "male")
cat(sprintf("%d jobs; ages %d-%d; %d draws\n", nrow(jobs), min(ages), max(ages), cfg$nsim))

# ── As in demography_forecasts.R ─────────────────────────────────────────────
as_matrix <- function(tab, p, s, years, column) {
    x <- tab[pop == p & sex == s & year %in% years]
    m <- matrix(NA_real_, length(ages), length(years))
    m[cbind(match(x$age, ages), match(x$year, years))] <- x[[column]]
    if (anyNA(m) && column != "rate") stop(sprintf("missing %s for %s %s", column, p, s))
    m
}

two_sex <- function(rate, pop, years, label) {
    dd <- demogdata(data = rate$female, pop = pop$female, ages = ages, years = years,
                    type = "mortality", label = label, name = "female")
    dd$rate$male <- rate$male
    dd$pop$male <- pop$male
    dd
}

failures <- list(); warns <- list()
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

# ── e0 by the chapter's life-table convention (mdmx.compare.e0_from_mx) ──────
# m: matrix of death rates, ages 0 … open age in rows, one schedule per column.
e0_chapter <- function(m) {
    m <- as.matrix(m)
    n_open <- nrow(m)
    m_full <- rbind(m[-n_open, , drop = FALSE], m[rep(n_open, cfg$n_ages - n_open + 1), , drop = FALSE])
    # a_x of the conversion: the book's a0 at age 0, cfg$ax above (mdmx.lifetable.conversion_ax).
    ax_age <- c(cfg$a0, rep(cfg$ax, nrow(m_full) - 1))
    q <- pmin(pmax(m_full / (1 + (1 - ax_age) * m_full), 0), 1)
    q <- pmin(pmax(q, cfg$q_min), 1 - cfg$q_min)
    z <- log(q) - log1p(-q)
    q <- 0.5 * (1 + tanh(0.5 * z))
    l <- rbind(1, apply(1 - q, 2, cumprod))                      # (n_ages + 1) × schedules
    if (!is.matrix(l)) l <- matrix(l, ncol = 1)
    L <- 0.5 * (l[-nrow(l), , drop = FALSE] + l[-1, , drop = FALSE])
    L[1, ] <- cfg$a0 * l[1, ] + (1 - cfg$a0) * l[2, ]
    colSums(L)
}

# e0 of an array of simulated rates (ages × h × nsim) → matrix h × nsim.
e0_draws <- function(sim) {
    d <- dim(sim)
    matrix(e0_chapter(matrix(sim, nrow = d[1])), nrow = d[2])
}

quant_rows <- function(method, job, sex, years_fc, e0) {
    q <- t(apply(e0, 1, quantile, probs = cfg$levels, names = FALSE, type = 7))
    data.table(method = method, pop = job$pop, origin = job$origin, sex = sex,
               year = rep(years_fc, times = length(cfg$levels)), level = rep(cfg$levels, each = length(years_fc)),
               e0 = as.vector(q))
}

point_rows <- function(method, job, sex, years_fc, rate, e0) {
    data.table(method = method, pop = job$pop, origin = job$origin, sex = sex, year = years_fc,
               e0_point = e0_chapter(rate), n_finite = rowSums(is.finite(e0)))
}

check_draws <- function(e0, what) {
    if (!all(is.finite(e0))) stop(sprintf("%s: %d non-finite e0 draws", what, sum(!is.finite(e0))))
}

quantiles <- list(); points <- list(); lc_kt <- list(); lc_age <- list()
started <- proc.time()[["elapsed"]]
for (i in seq_len(nrow(jobs))) {
    job <- jobs[i]
    years <- job$first_year:job$last_year
    years_fc <- job$last_year + seq_len(job$h)
    raw_rate <- raw_pop <- sm_rate <- list()
    for (s in sexes) {
        deaths <- as_matrix(rates, job$pop, s, years, "deaths")
        raw_pop[[s]] <- as_matrix(rates, job$pop, s, years, "exposure")
        raw_rate[[s]] <- ifelse(raw_pop[[s]] > 0, deaths / raw_pop[[s]], NA_real_)
        sm_rate[[s]] <- as_matrix(smooth[, .(pop, sex, year, age, rate = mx)], job$pop, s, years, "rate")
    }
    raw <- two_sex(raw_rate, raw_pop, years, job$pop)
    smoothed <- two_sex(sm_rate, raw_pop, years, job$pop)

    run_method("LC", job, {
        for (s in sexes) {
            fit <- lca(raw, series = s, max.age = max(ages), adjust = cfg$lc_adjust, interpolate = TRUE)
            fc <- forecast(fit, h = job$h, jumpchoice = cfg$lc_jumpchoice, level = 80)
            se <- (as.numeric(fc$kt.f$upper) - as.numeric(fc$kt.f$lower)) / (2 * qnorm(0.9))
            lc_kt[[length(lc_kt) + 1]] <- data.table(pop = job$pop, origin = job$origin, sex = s, year = years_fc,
                                                     kt_mean = as.numeric(fc$kt.f$mean), kt_se = se)
            lc_age[[length(lc_age) + 1]] <- data.table(pop = job$pop, origin = job$origin, sex = s, age = ages,
                                                       log_jump = log(fc$model$jumprates), bx = fit$bx)
            points[[length(points) + 1]] <- data.table(method = "LC", pop = job$pop, origin = job$origin,
                                                       sex = s, year = years_fc,
                                                       e0_point = e0_chapter(fc$rate[[s]]), n_finite = NA_integer_)
        }
    })

    run_method("HU", job, {
        e0s <- list()
        for (k in seq_along(sexes)) {
            s <- sexes[k]
            fit <- fdm(smoothed, series = s, order = cfg$fdm_order, method = cfg$fdm_method)
            fc <- forecast(fit, h = job$h, method = cfg$fdm_ts_method, jumpchoice = cfg$fdm_jumpchoice)
            set.seed(job$seed_hu + k - 1)
            e0s[[s]] <- e0_draws(simulate(fc, nsim = cfg$nsim))
            check_draws(e0s[[s]], sprintf("HU %s %s", job$pop, s))
            quantiles[[length(quantiles) + 1]] <- quant_rows("HU", job, s, years_fc, e0s[[s]])
            points[[length(points) + 1]] <- point_rows("HU", job, s, years_fc, fc$rate[[s]], e0s[[s]])
        }
        # The sexes are simulated separately: pairing draw i with draw i treats them as independent.
        quantiles[[length(quantiles) + 1]] <- quant_rows("HU", job, "both", years_fc,
                                                         0.5 * (e0s$female + e0s$male))
    })

    run_method("CFDM", job, {
        fit <- coherentfdm(smoothed, order1 = cfg$coherent_order_product, order2 = cfg$coherent_order_ratio)
        fc <- forecast(fit, h = job$h)
        set.seed(job$seed_cfdm)
        sim <- simulate(fc, nsim = cfg$nsim)
        e0s <- list()
        for (s in sexes) {
            e0s[[s]] <- e0_draws(sim[[s]])
            check_draws(e0s[[s]], sprintf("CFDM %s %s", job$pop, s))
            quantiles[[length(quantiles) + 1]] <- quant_rows("CFDM", job, s, years_fc, e0s[[s]])
            points[[length(points) + 1]] <- point_rows("CFDM", job, s, years_fc, fc[[s]]$rate[[s]], e0s[[s]])
        }
        # The two sexes share the product draw, so the per-draw average is CFDM's own joint distribution.
        quantiles[[length(quantiles) + 1]] <- quant_rows("CFDM", job, "both", years_fc,
                                                         0.5 * (e0s$female + e0s$male))
    })

    if (i %% 5 == 0 || i == nrow(jobs)) {
        cat(sprintf("  [%d/%d] %.0f s\n", i, nrow(jobs), proc.time()[["elapsed"]] - started))
    }
}

fwrite(rbindlist(quantiles), file.path(work, "quantiles.csv"))
fwrite(rbindlist(points), file.path(work, "points.csv"))
fwrite(rbindlist(lc_kt), file.path(work, "lc_kt.csv"))
fwrite(rbindlist(lc_age), file.path(work, "lc_age.csv"))
empty <- data.table(method = character(), pop = character(), origin = integer(), message = character())
fwrite(if (length(failures)) rbindlist(failures) else empty, file.path(work, "failures.csv"))
fwrite(if (length(warns)) rbindlist(warns) else empty, file.path(work, "warnings.csv"))
cat(sprintf("Done: %d failures, %d warnings\n", length(failures), length(warns)))
writeLines(capture.output(print(sessionInfo())), file.path(work, "sessionInfo.txt"))
