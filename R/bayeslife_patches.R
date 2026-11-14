# ══════════════════════════════════════════════════════════════════════════════
# bayeslife_patches.R — three changes inside bayesLife 5.3-1, sourced by bayeslife_forecast.R
# ══════════════════════════════════════════════════════════════════════════════
#
# Each function below copies one of bayesLife's internal functions and marks the changes "MDMx";
# patch_jmale() and patch_e0traj() install them in the bayesLife namespace of the calling R process
# (the main process and each chain's prediction worker).

# 1. The joint female–male gap model (Raftery, Lalic & Gerland 2014) for forecast origins before
# 1953: docs/bayeslife-issues.md BL-5 and BL-6.
#
# BL-5: bayesLife 5.3-1 hard-codes 1953 (female e0 in 1950–55, the start of the UN data) as the
# year of the gap model's covariate, in e0.jmale.estimate and .do.jmale.predict; with data that end
# before 1953 both fail.  `ref.year` is used instead: the earliest year in which every population of
# the run is observed (Sam's option B, 2026-10-28).
# BL-6: equation 1's term (e0F - 75)+ is not identified when (almost) no population-year has
# female e0 above 75, as before 1955; bayesLife's t-regression then fails.  The term is left out
# when fewer than `min.n.e0d75` population-years lie in (75, 83], and the forecast uses a coefficient
# of 0 (Sam's option (i), 2026-10-28: data up to the origin only).
#
# patch_jmale(ref.year) installs both changes in the bayesLife namespace of the calling R process;
# the chains' prediction workers call it too.  With ref.year = 1953 and the term identified, the
# estimator gives bayesLife's own estimates.

# bayesLife 5.3-1's e0.jmale.estimate, copied; the changes are marked "MDMx".
e0_jmale_estimate_mdmx <- function(mcmc.set, countries.index = NULL, estDof.eq1 = TRUE,
                                  start.eq1 = list(dof = 2), max.e0.eq1 = 83, estDof.eq2 = TRUE,
                                  start.eq2 = list(dof = 2), constant.gap.eq2 = TRUE,
                                  include.suppl.gap = FALSE, my.e0.file = NULL, my.locations.file = NULL,
                                  verbose = FALSE, ref.year = 1953, min.n.e0d75 = 10) {
    unblock.gtk("bDem.e0pred", list(bDem.e0pred.status = "estimating joint male"))
    if (is.null(countries.index))
        countries.index <- 1:get.nrest.countries(mcmc.set$meta)
    e0f.data <- get.data.matrix(mcmc.set$meta)[, countries.index]
    e0m.data.obj <- get.wpp.e0.data.for.countries(mcmc.set$meta, sex = "M", my.e0.file = my.e0.file,
                                                  my.locations.file = my.locations.file, verbose = verbose)
    e0m.data <- e0m.data.obj$e0.matrix[, countries.index]
    if (include.suppl.gap) {
        sdataF <- mcmc.set$meta$suppl.data$e0.matrix
        e0f.data <- rbind(matrix(NA, nrow = nrow(sdataF), ncol = ncol(e0f.data),
                                 dimnames = list(rownames(sdataF), colnames(e0f.data))), e0f.data)
        e0f.data[1:nrow(sdataF), colnames(sdataF)] <- sdataF
        sdataM <- e0m.data.obj$suppl.data$e0.matrix
        e0m.data <- rbind(matrix(NA, nrow = nrow(sdataM), ncol = ncol(e0m.data),
                                 dimnames = list(rownames(sdataM), colnames(e0m.data))), e0m.data)
        e0m.data[1:nrow(sdataM), colnames(sdataM)] <- sdataM
    }
    Tm <- dim(e0f.data)[1] - 1
    if (verbose) {
        cat("\nEstimating coefficients for joint female and male prediction.")
        cat("\nUsing", length(countries.index), "countries and", Tm + 1, "time periods.\n\n")
    }
    G <- e0f.data - e0m.data
    dep.var <- as.numeric(G[2:nrow(G), ])
    first.year <- max(ref.year, as.integer(rownames(e0f.data)[1]))           # MDMx: ref.year, not 1953
    if (first.year > ref.year)
        warning("Data for 1950-1955 not available. Estimation of the gap model may not be correct.")
    e0.1953 <- rep(e0f.data[as.character(first.year), ], each = Tm)
    e0F <- as.numeric(e0f.data[2:(Tm + 1), ])
    data <- data.frame(G = dep.var, e0.1953 = e0.1953, Gprev = as.numeric(G[1:Tm, ]), e0 = e0F,
                       e0d75 = pmax(0, e0F - 75))
    non.missing <- apply(data, 1, function(x) all(!is.na(x)))
    data.eq1 <- data[non.missing & e0F <= max.e0.eq1, ]
    drop.e0d75 <- sum(data.eq1$e0d75 > 0) < min.n.e0d75                      # MDMx: BL-6
    if (drop.e0d75) {
        warning("Fewer than ", min.n.e0d75, " data points with female e0 above 75: ",
                "the gap model's term (e0 - 75)+ is left out (coefficient 0).")
        data.eq1$e0d75 <- NULL
    }
    fit.eq1 <- tlm(G ~ ., data = data.eq1, estDof = estDof.eq1, start = start.eq1)
    if (verbose) {
        cat("\n\nUsing", nrow(data.eq1), "data points for equation 1.\n\n")
        print(summary(fit.eq1))
    }
    errscale.eq1 <- as.numeric(exp(fit.eq1$scale.fit$coefficients[1]))
    errsd.eq1 <- sqrt(errscale.eq1)
    data.eq2 <- data[non.missing & e0F > max.e0.eq1, ]
    if (verbose)
        cat("\n\nUsing", nrow(data.eq2), "data points for equation 2.\n\n")
    if (!constant.gap.eq2) {
        fit.eq2 <- tlm(G ~ -1 + Gprev, data = data.eq2, start = start.eq2, estDof = estDof.eq2)
        if (verbose) print(summary(fit.eq2))
        errscale.eq2 <- as.numeric(exp(fit.eq2$scale.fit$coefficients[1]))
        errsd.eq2 <- sqrt(errscale.eq2)
        coef2 <- fit.eq2$loc.fit$coefficients
        dof2 <- fit.eq2$dof
    } else {
        dof2 <- NULL
        coef2 <- c(Gprev = 1)
        errsd.eq2 <- sqrt(mean((data.eq2$G - data.eq2$Gprev)^2))
    }
    coef1 <- fit.eq1$loc.fit$coefficients
    if (drop.e0d75) coef1 <- c(coef1, e0d75 = 0)                               # MDMx: BL-6
    return(list(eq1 = list(coefficients = coef1, sigma = errsd.eq1, dof = fit.eq1$dof),
                eq2 = list(coefficients = coef2, sigma = errsd.eq2, dof = dof2)))
}

patch_jmale <- function(ref.year) {
    ns <- asNamespace("bayesLife")
    est <- e0_jmale_estimate_mdmx
    formals(est)$ref.year <- ref.year
    environment(est) <- ns
    utils::assignInNamespace("e0.jmale.estimate", est, ns = "bayesLife")
    # The male prediction needs only the year: rewrite the literal 1953 (not the name "e0.1953").
    # Always from bayesLife's own function, kept on the patched one: a process may be patched more
    # than once (a prediction worker serves the convergence check and the final forecast).
    cur <- get(".do.jmale.predict", envir = ns)
    orig <- if (is.null(attr(cur, "mdmx_original"))) cur else attr(cur, "mdmx_original")
    src <- deparse(orig)
    new <- gsub("(?<![.[:alnum:]_])1953(?![[:alnum:]_])", as.character(ref.year), src, perl = TRUE)
    if (identical(new, src) && ref.year != 1953) stop("patch_jmale: no 1953 in .do.jmale.predict")
    f <- eval(parse(text = new), envir = ns)
    environment(f) <- ns
    attr(f, "mdmx_original") <- orig
    utils::assignInNamespace(".do.jmale.predict", f, ns = "bayesLife")
}


# 2. Female e0 trajectories bounded below: docs/bayeslife-issues.md BL-16.
#
# bayesLife 5.3-1's generate.e0.trajectory adds a normal error to every year's step with no bound.
# From early origins its noise scale ω is large (2.4–2.7), so over 50 years a few trajectories fall
# below e0 0; the spread look-up loess.lookup() returns NA there, the next step's g.dl6 fails and
# the whole forecast is lost.  A year's error is redrawn when it would take e0 below `lower`
# (0), up to `max.tries` draws in all, and sets e0 to the bound if every draw fails — the rule of
# bayesTFR's make.tfr.prediction for TFR below min.tfr, and of bayesLife's own gap model for gaps out
# of range (Sam's option (b), 2026-10-29).  The redraws come from the same random stream, so a run in
# which no trajectory goes below the bound is bit-identical to bayesLife's.
#
# patch_e0traj() installs the generator in the bayesLife namespace of the calling R process and
# returns the environment that counts its redraws and clamps; the chains' prediction workers call
# it too (they source this file).

# bayesLife 5.3-1's generate.e0.trajectory, copied; the changes are marked "MDMx".  `...` takes
# the arguments bayesLife passes and does not use (traj, pred.env).
generate_e0_trajectory_mdmx <- function(x, l.start, kap, n.proj = 11, p1 = 9, p2 = 9, const.var = FALSE,
                                       ..., lower = 0, max.tries = 50) {
    proj <- NULL
    proj[1] <- l.start
    for (a in 2:(n.proj + 1)) {
        # Evaluated in bayesLife's order — (previous + gain) + error — so the sums are its own.
        step <- proj[a - 1] + g.dl6(x, proj[a - 1], p1 = p1, p2 = p2)
        err.sd <- kap * if (const.var) 1 else loess.lookup(proj[a - 1])
        proj[a] <- step + rnorm(1, mean = 0, sd = err.sd)
        tries <- 1                                                            # MDMx: BL-16
        while (proj[a] < lower && tries < max.tries) {
            proj[a] <- step + rnorm(1, mean = 0, sd = err.sd)
            tries <- tries + 1
        }
        if (tries > 1) .bl16$redraws <- .bl16$redraws + 1
        if (proj[a] < lower) {
            proj[a] <- lower
            .bl16$clamped <- .bl16$clamped + 1
        }
    }
    return(proj[2:length(proj)])
}

patch_e0traj <- function(lower = 0, max.tries = 50) {
    ns <- asNamespace("bayesLife")
    # The counters live in an environment between the function and bayesLife's namespace, so the
    # generator finds both them and bayesLife's internals (g.dl6, loess.lookup).
    env <- new.env(parent = ns)
    env$.bl16 <- new.env()
    env$.bl16$redraws <- 0
    env$.bl16$clamped <- 0
    f <- generate_e0_trajectory_mdmx
    formals(f)$lower <- lower
    formals(f)$max.tries <- max.tries
    environment(f) <- env
    utils::assignInNamespace("generate.e0.trajectory", f, ns = "bayesLife")
    env$.bl16
}
