# ══════════════════════════════════════════════════════════════════════════════
# bayeslife_forecast.R — the UN's e0 model (bayesLife) on HMD data, one origin
# ══════════════════════════════════════════════════════════════════════════════
#
# Used by book chapter 12-evaluation (paper §12, decision D2).  For one
# forecast origin and one fitting-period policy, bayesLife 5.3-1 (Raftery,
# Chunn, Gerland & Ševčíková 2013) is estimated on HMD life expectancies only
# and projects female e0 by its Bayesian hierarchical double-logistic model
# and male e0 by its joint female–male gap model.
#
#   * HMD data only: use.wpp.data = FALSE, so the user file replaces the WPP
#     dataset completely; every population has its own numeric code (10001…)
#     and a row in the locations file.
#   * The world-level parameters are estimated from non-overlapping
#     populations only (include_code 2; Sam, 2026-10-26); the parts of an
#     aggregate that is itself in the run are added afterwards by
#     run.e0.mcmc.extra, which fits their own parameters given the world
#     posterior without changing it.
#   * Convergence is judged on the forecasts (Sam, 2026-10-27): the
#     likelihood has ridges — the k / lambda.k block of the world level mixes
#     very slowly — so the chains (bayesLife's random starting values) are
#     extended in steps of `iter_incr` until (i) every world parameter except
#     k and lambda.k has a Gelman–Rubin R-hat of at most `rhat_max`, and
#     (ii) the forecasts agree: each chain predicts every population's female
#     e0 at the horizon with the *same* random numbers (so that the chains
#     differ only through their parameters, not through the simulated future)
#     and the largest difference between chains is at most `tol_median` years
#     for the median and `tol_bounds` for the 2.5 % and 97.5 % points — or
#     until `max_loops` is reached.  A forecast R-hat would not do: the
#     simulated future dominates the spread, so it tolerates chains whose
#     centres differ by most of a year.  Every loop's diagnostics, with
#     bayesLife's Raftery–Lewis check, are written to diagnostics.csv.
#   * Every prediction — each chain's check and the final forecast — uses
#     every stored draw after burn-in (bayesLife's default is 2,000), so that
#     its Monte-Carlo error falls as the chains grow; with a fixed number of
#     trajectories the chains' forecasts would differ by sampling alone, however
#     long they ran (Sam, 2026-10-27).  The chains predict in parallel, the
#     final forecast chain by chain with its own seed, pooled.
#   * The gap model's equation 2 (female e0 above 83) has no data before
#     1996; where its σ cannot be estimated, bayesLife's own WPP-derived
#     annual value is used and the substitution is recorded (Sam, 2026-10-26).
#   * The chains' forecasts are compared at `check_year` (origin + 15), while
#     the forecast runs to `end_year` (origin + 50).
#   * Three changes inside bayesLife, installed by bayeslife_patches.R
#     (docs/bayeslife-issues.md):
#     - BL-5 and BL-6: the gap model before 1953;
#     - BL-16: female e0 trajectories bounded below by `e0_lower`.  A year's
#       error that would take e0 below the bound is redrawn.  The redraws and
#       clamps per chain are written to redraws.csv.  A run in which no
#       trajectory reaches the bound is bit-identical to bayesLife's.
#   * `jmale = false` forecasts female e0 only.
#
# No error is ignored: a failure stops the run (R exits with status 1 after
# writing failures.csv); every warning is written to warnings.csv.
#
# Called from Python through mdmx.rbridge:
#     Rscript --vanilla R/bayeslife_forecast.R <work folder>
#
# Input   female.csv    country_code, name, include_code, <year> … <origin>  (e0, NA = no data)
#         male.csv      the same columns for male e0
#         locations.csv country_code, name, location_type
#         settings.json origin, end_year, check_year, start_year, extra (codes), sim_dir,
#                       nr_chains, iter, thin, burnin, iter_incr, max_loops, rhat_max,
#                       tol_median, tol_bounds, seed, wpp_year, diag_thin, jmale,
#                       e0_lower, e0_max_tries
# Output  forecasts.csv country_code, sex, year, median, q10, q90, q025, q975
#         diagnostics.csv kind (world/forecast/final), loop, iter, name (parameter, population
#                       code or "trajectories"), rhat, ess, min_distinct, median_spread,
#                       lower_spread, upper_spread, n_traj (per chain in a check; pooled in
#                       the final row), raftery_ok, raftery_iter_needed, note, converged
#         gap.csv       the joint-male estimates and whether eq. 2's σ was substituted
#         redraws.csv   step (check <loop> / final), chain, redraws, clamped (BL-16: trajectory-
#                       years whose error was redrawn, and those set to the bound)
#         lowtail.csv   country_code, n_traj, below20_any, below20_end: the shares of the final
#                       female trajectories that fall below e0 20 in some year, and at the end year
#         failures.csv, warnings.csv, timing.csv, sessionInfo.txt
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
    library(bayesLife)
    library(coda)
    library(data.table)
    library(jsonlite)
})

work <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(work)) stop("Usage: Rscript bayeslife_forecast.R <work folder>")
cfg <- fromJSON(file.path(work, "settings.json"))
# The simulation lives outside the work folder (which rbridge empties), so that a run can later be
# extended or predicted again without being redone.
sim.dir <- cfg$sim_dir
warns <- list()
timing <- list()
bl16 <- list()
stamp <- function(step, t0) timing[[length(timing) + 1]] <<- data.table(step = step,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))

# bayesLife reads tab-separated files (read.delim) and treats "#" as a comment.
to_tsv <- function(name) {
    d <- fread(file.path(work, paste0(name, ".csv")), colClasses = list(character = "name"))
    if (any(grepl("#", d$name))) stop("a population name contains '#'")
    path <- file.path(work, paste0(name, ".tsv"))
    fwrite(d, path, sep = "\t", na = "NA", quote = TRUE)
    path
}
female <- to_tsv("female")
male <- to_tsv("male")
locs <- to_tsv("locations")
fem <- fread(female)
all_codes <- fem$country_code
main_codes <- fem[include_code == 2, country_code]

# bayesLife's joint female–male gap model (Raftery, Lalic & Gerland 2014) uses each population's
# female e0 in 1953 — the start of the UN data, 1950–55 — as a covariate: e0.jmale.estimate takes
# the year max(1953, first year of the data), .do.jmale.predict max(1953, the population's first
# observed year).  With data that end before 1953 both fail ("subscript out of bounds").  The
# earliest year in which every population of the run is observed is used instead: the same role, the
# start of the common observation period.  With 1953 inside the data nothing changes (ref_year NULL).
# docs/bayeslife-issues.md, issue BL-5.
year_cols <- setdiff(names(fem), c("country_code", "name", "include_code"))
first_obs <- apply(as.matrix(fem[, ..year_cols]), 1, function(v) as.integer(year_cols[which(!is.na(v))[1]]))
ref_year <- if (cfg$origin < 1953) max(first_obs) else NULL

# patch_jmale() and patch_e0traj(): BL-5, BL-6 and BL-16 (bayeslife_patches.R, beside this file).
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "bayeslife_patches.R"))
# The patches every prediction installs, in this process and in the chains' workers.  A worker
# receives functions without the script's globals, so it sources the files itself.
if (is.null(cfg$e0_lower) || is.null(cfg$e0_max_tries)) stop("settings need e0_lower and e0_max_tries (BL-16)")
patches <- list(files = file.path(script_dir, "bayeslife_patches.R"), ref.year = ref_year,
                e0.lower = cfg$e0_lower, max.tries = cfg$e0_max_tries)
parallel <- cfg$nr_chains > 1
chains <- seq_len(cfg$nr_chains)
cl <- NULL                                         # the cluster of the chains' predictions

# One chain's prediction, in its own R process: the cluster serialises this function without the
# script's globals, so every input is an argument.  `nr.traj` = every stored draw after burn-in.
# A worker's warnings never reach the main process, so they are returned and raised there; so are
# its BL-16 counts.
predict_chain <- function(ch, sim.dir, out.dirs, end.year, burnin, seeds, fixes, jmale = NULL) {
    warns <- character(0)
    counts <- NULL
    n <- withCallingHandlers({
        suppressPackageStartupMessages(library(bayesLife))
        env <- new.env()
        for (f in fixes$files) sys.source(f, envir = env)
        if (!is.null(fixes$ref.year)) env$patch_jmale(fixes$ref.year)
        counts <- env$patch_e0traj(fixes$e0.lower, fixes$max.tries)
        mc <- get.e0.mcmc(sim.dir, chain.ids = ch)
        n <- bayesTFR::get.stored.mcmc.length(mc$mcmc.list, burnin)
        p <- e0.predict(mcmc.set = mc, end.year = end.year, burnin = burnin, nr.traj = n,
                        predict.jmale = FALSE, replace.output = TRUE, output.dir = out.dirs[ch],
                        seed = seeds[ch], verbose = FALSE)
        if (!is.null(jmale))
            e0.jmale.predict(p, estimates = jmale$estimates, my.e0.file = jmale$male,
                             my.locations.file = jmale$locs, verbose = FALSE)
        n
    }, warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    list(n_traj = n, warnings = warns, redraws = counts$redraws, clamped = counts$clamped)
}

on_chains <- function(step, ...) {
    res <- if (is.null(cl)) lapply(chains, predict_chain, ...) else parallel::parLapply(cl, chains, predict_chain, ...)
    for (r in res) for (m in r$warnings) warning(m, call. = FALSE)
    bl16[[length(bl16) + 1]] <<- data.table(step = step, chain = chains,
                                            redraws = vapply(res, `[[`, numeric(1), "redraws"),
                                            clamped = vapply(res, `[[`, numeric(1), "clamped"))
    n <- vapply(res, `[[`, numeric(1), "n_traj")
    if (length(unique(n)) != 1) stop("the chains have different numbers of stored draws: ", toString(n))
    n[1]
}

# Gelman–Rubin and effective sample size of the world-level parameters, and
# bayesLife's Raftery–Lewis check, on the stored (thinned) chains after burn-in.
convergence <- function(loop) {
    mc <- get.e0.mcmc(sim.dir)
    cl <- e0.coda.list.mcmc(mc, par.names = e0.parameter.names(), par.names.cs = NULL,
                            burnin = cfg$burnin)
    g <- gelman.diag(cl, autoburnin = FALSE, multivariate = FALSE)$psrf[, 1]
    ess <- effectiveSize(cl)
    # A parameter that keeps the same value through a chain is a stuck sampler (the likelihood has
    # ridges); count its distinct values in the worst chain.
    distinct <- apply(sapply(cl, function(ch) apply(ch, 2, function(v) length(unique(v)))), 1, min)
    # Raftery–Lewis on the world and the population parameters, as bayesLife's own "auto" mode does.
    # bayesLife drops parameters that are constant within a chain and then fails to combine the
    # chains, and cannot diagnose chains shorter than the method's minimum: both are recorded as
    # "not converged", with the reason, never ignored.  (The check is reported; it does not stop
    # the chains — the forecasts do.)
    rl <- tryCatch(e0.diagnose(sim.dir = sim.dir, thin = cfg$diag_thin, burnin = cfg$burnin,
                               express = FALSE, verbose = FALSE),
                   error = function(e) {
                       msg <- conditionMessage(e)
                       note <- if (grepl("Different number of variables", msg)) {
                           "a parameter is constant within a chain"
                       } else if (grepl("Problem in e0.raftery.diag", msg)) {
                           "Raftery-Lewis could not be computed (too few draws)"
                       } else stop(e)
                       list(status = c(red = TRUE, green = FALSE), iter.needed = NA_real_, note = note)
                   })
    world <- data.table(kind = "world", loop = loop, iter = mc$mcmc.list[[1]]$finished.iter,
                        name = names(g), rhat = as.numeric(g), ess = as.numeric(ess[names(g)]),
                        min_distinct = as.integer(distinct[names(g)]), median_spread = NA_real_,
                        raftery_ok = isTRUE(rl$status[["green"]]), raftery_iter_needed = rl$iter.needed,
                        note = if (is.null(rl$note)) "" else rl$note)
    # The forecasts, chain by chain, with common random numbers (one seed for every chain): female
    # e0 at the horizon of every population of the world fit.  The predictions are only needed for
    # these numbers, so they are deleted afterwards (each holds a thinned copy of its chain).
    pdirs <- file.path(sim.dir, "check", paste0("chain", chains))
    n_traj <- on_chains(paste("check", loop), sim.dir = sim.dir, out.dirs = pdirs, end.year = check_year,
                        burnin = cfg$burnin, seeds = rep(cfg$seed, length(chains)), fixes = patches)
    ends <- lapply(pdirs, function(pdir) {
        p <- get.e0.prediction(pdir)
        sapply(main_codes, function(code) get.e0.trajectories(p, code)[as.character(check_year), ])
    })                                                                    # trajectories × populations
    unlink(file.path(sim.dir, "check"), recursive = TRUE)
    spread <- function(j, prob) diff(range(sapply(ends, function(e) quantile(e[, j], prob, names = FALSE))))
    fc <- rbindlist(lapply(seq_along(main_codes), function(j) {
        data.table(kind = "forecast", loop = loop, iter = mc$mcmc.list[[1]]$finished.iter,
                   name = as.character(main_codes[j]), rhat = NA_real_, ess = NA_real_,
                   min_distinct = NA_integer_, median_spread = spread(j, 0.5),
                   lower_spread = spread(j, 0.025), upper_spread = spread(j, 0.975), n_traj = n_traj,
                   raftery_ok = NA, raftery_iter_needed = NA_real_, note = "")
    }))
    rbind(world, fc, fill = TRUE)
}

# A NaN (e.g. an R-hat of a parameter that is constant in every chain) counts as not converged.
converged <- function(d) {
    ok_world <- d[kind == "world" & !(name %in% c("k", "lambda.k")), isTRUE(all(rhat <= cfg$rhat_max))]
    ok_fc <- d[kind == "forecast", isTRUE(all(median_spread <= cfg$tol_median & lower_spread <= cfg$tol_bounds &
                                                  upper_spread <= cfg$tol_bounds))]
    ok_world && ok_fc
}

# The year at which the chains' forecasts must agree (default: the end of the forecast).
check_year <- if (is.null(cfg$check_year)) cfg$end_year else cfg$check_year

result <- withCallingHandlers(tryCatch({
    t0 <- Sys.time()
    run.e0.mcmc(sex = "F", nr.chains = cfg$nr_chains, iter = cfg$iter, output.dir = sim.dir,
                thin = cfg$thin, replace.output = TRUE, annual = TRUE, start.year = cfg$start_year,
                present.year = cfg$origin, wpp.year = cfg$wpp_year, my.e0.file = female,
                my.locations.file = locs, use.wpp.data = FALSE, seed = cfg$seed,
                parallel = parallel, nr.nodes = cfg$nr_chains, verbose = FALSE)
    stamp("mcmc", t0)
    if (parallel) cl <- parallel::makePSOCKcluster(cfg$nr_chains)
    diag <- list()
    for (loop in seq_len(cfg$max_loops)) {
        t0 <- Sys.time()
        d <- convergence(loop)
        stamp(paste("check", loop), t0)
        diag[[loop]] <- d
        if (converged(d)) break
        if (loop == cfg$max_loops) break
        t0 <- Sys.time()
        continue.e0.mcmc(iter = cfg$iter_incr, output.dir = sim.dir, parallel = parallel,
                         nr.nodes = cfg$nr_chains, verbose = FALSE)
        stamp(paste("continue", loop), t0)
    }
    diag <- rbindlist(diag)
    if (length(cfg$extra) > 0) {
        t0 <- Sys.time()
        run.e0.mcmc.extra(sim.dir = sim.dir, countries = cfg$extra, my.e0.file = female,
                          my.locations.file = locs, burnin = cfg$burnin, parallel = parallel,
                          nr.nodes = cfg$nr_chains, verbose = FALSE)
        stamp("extra", t0)
    }
    # The final forecast: every chain predicts female e0 from all its stored draws, with its own
    # seed, and male e0 through the gap model; the trajectories of the chains are pooled.
    # `jmale = FALSE`: female e0 only.
    jmale <- if (is.null(cfg$jmale)) TRUE else isTRUE(cfg$jmale)
    t0 <- Sys.time()
    if (jmale) {
        if (!is.null(ref_year)) patch_jmale(ref_year)
        est <- bayesLife:::e0.jmale.estimate(get.e0.mcmc(sim.dir), my.e0.file = male, my.locations.file = locs)
        substituted <- !is.finite(est$eq2$sigma)
        if (substituted) est$eq2$sigma <- bayesLife:::subnat.gap.estimates(annual = TRUE)$eq2$sigma
    }
    fdirs <- file.path(sim.dir, "final", paste0("chain", chains))
    n_final <- on_chains("final", sim.dir = sim.dir, out.dirs = fdirs, end.year = cfg$end_year,
                         burnin = cfg$burnin, seeds = cfg$seed + chains, fixes = patches,
                         jmale = if (jmale) list(estimates = est, male = male, locs = locs)) * length(chains)
    preds <- lapply(fdirs, get.e0.prediction)
    mpreds <- if (jmale) lapply(preds, get.e0.jmale.prediction)
    stamp("predict", t0)
    pooled <- function(ps, code) {
        tr <- lapply(ps, get.e0.trajectories, code)                  # years × trajectories, per chain
        if (any(vapply(tr, is.null, logical(1)))) stop("no trajectories for population ", code)
        do.call(cbind, tr)
    }
    probs <- c(median = 0.5, q10 = 0.1, q90 = 0.9, q025 = 0.025, q975 = 0.975)
    rows <- list()
    low <- list()
    for (code in all_codes) {
        fT <- pooled(preds, code)                                  # years × trajectories
        if (ncol(fT) != n_final) stop("population ", code, " has ", ncol(fT), " trajectories, not ", n_final)
        fy <- as.integer(rownames(fT))
        fK <- fT[fy > cfg$origin & fy <= cfg$end_year, , drop = FALSE]
        low[[length(low) + 1]] <- data.table(country_code = code, n_traj = ncol(fK),
                                             below20_any = mean(apply(fK, 2, min) < 20),
                                             below20_end = mean(fK[nrow(fK), ] < 20))
        if (jmale) {
            mT <- pooled(mpreds, code)
            if (!identical(dim(fT), dim(mT))) stop("female and male trajectories differ in shape for ", code)
        }
        yrs <- as.integer(rownames(fT))
        keep <- yrs > cfg$origin & yrs <= cfg$end_year
        for (sex in if (jmale) c("female", "male", "both") else "female") {
            X <- switch(sex, female = fT, male = mT, both = (fT + mT) / 2)[keep, , drop = FALSE]
            q <- t(apply(X, 1, quantile, probs = probs, names = FALSE))
            colnames(q) <- names(probs)
            rows[[length(rows) + 1]] <- data.table(country_code = code, sex = sex, year = yrs[keep], q)
        }
    }
    gap <- if (jmale) data.table(equation = c("eq1", "eq2"), sigma = c(est$eq1$sigma, est$eq2$sigma),
                                 substituted = c(FALSE, substituted),
                                 ref_year = if (is.null(ref_year)) 1953L else as.integer(ref_year)) else
        data.table(equation = character(0), sigma = numeric(0), substituted = logical(0), ref_year = integer(0))
    # The trajectories are regenerable from the chains and the seeds; only their quantiles are kept.
    unlink(file.path(sim.dir, "final"), recursive = TRUE)
    if (!is.null(cl)) parallel::stopCluster(cl)
    final <- data.table(kind = "final", loop = max(diag$loop), iter = max(diag$iter), name = "trajectories",
                        n_traj = n_final)
    diag <- rbind(diag, final, fill = TRUE)
    diag[, converged := converged(diag[loop == max(loop)])]
    list(forecasts = rbindlist(rows), diagnostics = diag, gap = gap, lowtail = rbindlist(low))
}, error = function(e) {
    if (!is.null(cl)) parallel::stopCluster(cl)
    fwrite(data.table(message = conditionMessage(e)), file.path(work, "failures.csv"))
    if (length(bl16)) fwrite(rbindlist(bl16), file.path(work, "redraws.csv"))
    fwrite(rbindlist(warns), file.path(work, "warnings.csv"))
    writeLines(capture.output(sessionInfo()), file.path(work, "sessionInfo.txt"))
    quit(status = 1)
}), warning = function(w) {
    warns[[length(warns) + 1]] <<- data.table(message = conditionMessage(w))
    invokeRestart("muffleWarning")
})

fwrite(result$forecasts, file.path(work, "forecasts.csv"))
fwrite(result$diagnostics, file.path(work, "diagnostics.csv"))
fwrite(result$gap, file.path(work, "gap.csv"))
fwrite(data.table(message = character(0)), file.path(work, "failures.csv"))
fwrite(if (length(warns)) rbindlist(warns) else data.table(message = character(0)),
       file.path(work, "warnings.csv"))
fwrite(rbindlist(timing), file.path(work, "timing.csv"))
fwrite(rbindlist(bl16), file.path(work, "redraws.csv"))
fwrite(result$lowtail, file.path(work, "lowtail.csv"))
writeLines(capture.output(sessionInfo()), file.path(work, "sessionInfo.txt"))
