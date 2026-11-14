# ══════════════════════════════════════════════════════════════════════════════
# Install the R packages used by the MDMx forecasting comparators.
# Run once:   Rscript R/install-packages.R
#
# Strategy: install the current CRAN *binary* of each package (fast, no
# compilers needed on macOS/Windows), then check its version against the
# version used for the paper.  Only if they differ is the pinned version built
# from source with remotes::install_version (which needs Xcode command-line
# tools + gfortran on macOS, or Rtools on Windows).
#
# What is pinned: the whole dependency closure (Depends, Imports and LinkingTo,
# recursively) of the packages the book's R scripts use, each at the version
# that ran.  The versions are those of the sessionInfo() records of every R
# step of the book (work/r/sessionInfo-*.txt: demography_forecasts,
# demography_intervals, hmd_smooth, mortcast_schedules, bayeslife_forecast)
# and of chapter 12's 48 bayesLife runs (results/*/sessionInfo.txt of the
# off-site project), all under R 4.5.3, checked against the R 4.5.3 library
# that ran them (2026-11-01).  Two kinds of package in the closure were loaded
# by no run, and are pinned at the version of that library:
#   "not loaded"   installed only because a package lists it;
#   "build only"   C/C++ headers used when a package is compiled (LinkingTo).
# The M4's 18 bayesLife runs (origins 1900–1945) loaded newer versions of four
# packages, marked below; every other run used the versions pinned here.
# The list is in dependency order, so that a version built from source is built
# against its own pinned dependencies.
# ══════════════════════════════════════════════════════════════════════════════
options(repos = c(CRAN = "https://cloud.r-project.org"))

# The R version every run used.  Another version works, but may not reproduce
# the runs bit for bit, so it only warns.
r_version <- "4.5.3"
if (as.character(getRversion()) != r_version)
  warning("This is R ", getRversion(), "; the book's R steps ran under R ", r_version,
          ": results may not be reproduced bit for bit.", call. = FALSE, immediate. = TRUE)

pinned <- c(
  abind = "1.4-8",
  ash = "1.0-15",                 # not loaded
  backports = "1.5.0",
  base64enc = "0.1-6",
  bit = "4.6.0",                  # not loaded
  bitops = "1.0-9",
  boot = "1.3-32",
  carData = "3.0-6",
  cli = "3.6.5",                  # the M4's 18 bayesLife runs: 3.6.6
  clipr = "0.8.0",                # not loaded
  cluster = "2.1.8.2",
  colorspace = "2.1-2",
  cpp11 = "0.5.3",                # build only (LinkingTo)
  crayon = "1.5.3",               # not loaded
  curl = "7.0.0",                 # not loaded
  data.table = "1.18.6.1",
  Deriv = "4.2.0",                # not loaded
  deSolve = "1.41",
  digest = "0.6.39",
  evaluate = "1.0.5",
  farver = "2.1.2",
  fastICA = "1.2-7",              # not loaded
  fastmap = "1.2.0",
  FNN = "1.1.4.1",                # not loaded
  foreign = "0.8-91",
  Formula = "1.2-5",              # the M4's 18 bayesLife runs: 1.2-6
  fracdiff = "1.5-3",
  fs = "1.6.7",                   # not loaded
  generics = "0.1.4",
  glue = "1.8.0",
  GPArotation = "2025.3-1",       # not loaded
  gss = "2.2-10",                 # not loaded
  jsonlite = "2.0.0",
  kernlab = "0.9-33",             # not loaded
  KernSmooth = "2.23-26",
  labeling = "0.4.3",             # not loaded
  LaplacesDemon = "16.1.8",
  lattice = "0.22-9",
  longmemo = "1.1-4",
  lpSolve = "5.6.23",             # not loaded
  magrittr = "2.0.4",
  MASS = "7.3-65",
  mclust = "6.1.2",
  microbenchmark = "1.5.0",       # not loaded
  mime = "0.13",                  # not loaded
  mnormt = "2.1.2",
  mvtnorm = "1.3-5",              # the M4's 18 bayesLife runs: 1.4-2
  nloptr = "2.2.1",               # not loaded
  nnet = "7.3-20",
  numDeriv = "2016.8-1.1",
  otel = "0.2.0",
  pkgconfig = "2.0.3",
  pracma = "2.4.6",
  prettyunits = "1.2.0",          # build only (LinkingTo)
  proxy = "0.4-29",
  R6 = "2.6.1",
  rappdirs = "0.3.4",             # not loaded
  rbibutils = "2.4.1",
  RColorBrewer = "1.1-3",
  Rcpp = "1.1.1",
  RcppProgress = "0.4.2",         # build only (LinkingTo)
  rlang = "1.1.7",                # the M4's 18 bayesLife runs: 1.3.0
  rlecuyer = "0.3-8",
  rpart = "4.1.24",
  rstudioapi = "0.18.0",
  S7 = "0.2.1",
  snow = "0.4-4",
  SparseM = "1.84-2",
  spatial = "7.3-18",
  stabledist = "0.7-2",           # not loaded
  stringi = "1.8.7",
  sys = "3.4.3",                  # not loaded
  timeDate = "4052.112",
  utf8 = "1.2.6",                 # not loaded
  viridisLite = "0.4.3",          # not loaded
  withr = "3.0.2",                # not loaded
  xfun = "0.56",
  yaml = "2.3.12",                # not loaded
  askpass = "1.2.1",              # not loaded
  bit64 = "4.6.0-1",              # not loaded
  cachem = "1.1.0",               # not loaded
  checkmate = "2.3.4",
  class = "7.3-23",
  coda = "0.19-4.1",              # R-hat, effective sample size (bayeslife_forecast.R)
  ecp = "3.1.6",
  hett = "0.3-3",                 # bayesLife's gap model (t regression, tlm)
  highr = "0.12",                 # not loaded
  htmltools = "0.5.9",
  isoband = "0.3.0",              # not loaded
  lifecycle = "1.0.5",
  linprog = "0.9-6",              # not loaded
  locfit = "1.5-9.12",            # not loaded
  magic = "1.6-1",
  Matrix = "1.7-4",
  minqa = "1.2.8",                # not loaded
  multicool = "1.0.1",            # not loaded
  nlme = "3.1-168",
  pcaPP = "2.0-5",
  RcppArmadillo = "15.2.3-1",     # build only (LinkingTo)
  RcppEigen = "0.3.4.0.2",        # build only (LinkingTo)
  RCurl = "1.98-1.17",
  Rdpack = "2.6.6",
  snowFT = "1.6-1",               # bayesLife's parallel chains (parallel = TRUE)
  timechange = "0.4.0",           # not loaded
  timeSeries = "4052.112",
  tinytex = "0.58",               # not loaded
  tzdb = "0.5.0",                 # not loaded
  wpp2017 = "1.2-3",
  wpp2019 = "1.1-1",              # bayesLife's location table and constants (wpp.year = 2019)
  xml2 = "1.5.2",                 # not loaded
  zoo = "1.8-15",
  bayesTFR = "7.4-5",
  e1071 = "1.7-17",
  fBasics = "4052.98",
  fontawesome = "0.5.3",          # not loaded
  gbutils = "0.5.1",
  geometry = "0.5.2",
  gtable = "0.3.6",
  jquerylib = "0.1.4",            # not loaded
  knitr = "1.51",
  Lmoments = "1.3-2",
  lmtest = "0.9-40",
  lubridate = "1.9.5",            # not loaded
  MatrixModels = "0.5-4",
  memoise = "2.0.1",              # not loaded
  mgcv = "1.9-4",
  MortCast = "2.8-0",
  openssl = "2.3.5",              # not loaded
  psych = "2.6.1",
  reformulas = "0.4.4",           # not loaded
  sandwich = "3.1-1",
  sass = "0.4.10",                # not loaded
  scales = "1.4.0",
  survival = "3.8-6",
  urca = "1.3-4",
  vctrs = "0.7.1",
  bslib = "0.10.0",               # not loaded
  cvar = "0.6",
  evgam = "1.0.1",
  ggplot2 = "4.0.0",
  gridExtra = "2.3",
  hms = "1.1.4",                  # not loaded
  httr = "1.4.8",                 # not loaded
  ks = "1.15.1",
  lme4 = "2.0-1",                 # not loaded
  pdfCluster = "1.0-4",
  pillar = "1.11.1",
  purrr = "1.2.1",                # not loaded
  quantreg = "6.1",
  ROOPSD = "0.3.9",
  stringr = "1.6.0",
  strucchange = "1.5-4",
  tidyselect = "1.2.1",
  cobs = "1.3-9-1",
  cowplot = "1.2.0",              # not loaded
  fGarch = "4052.93",
  forecast = "9.0.2",             # ARIMA for the functional-data models
  hdrcde = "3.5.0",
  progress = "1.2.3",             # build only (LinkingTo)
  rmarkdown = "2.30",
  selectr = "0.5-1",              # not loaded
  snakecase = "0.11.1",           # not loaded
  tibble = "3.3.1",
  vars = "1.6-1",
  dplyr = "1.2.0",
  htmlwidgets = "1.6.4",
  rainbow = "3.8",
  rvest = "1.0.5",                # not loaded
  vroom = "1.7.0",                # not loaded
  fds = "1.8",
  htmlTable = "2.4.3",
  readr = "2.2.0",                # not loaded
  tidyr = "1.3.2",                # not loaded
  broom = "1.0.12",               # not loaded
  fda = "6.3.0",
  Hmisc = "5.2-5",
  janitor = "2.2.1",              # not loaded
  fdapace = "0.6.0",
  HMDHFDplus = "2.0.8",           # not loaded
  modelr = "0.1.11",              # not loaded
  sde = "2.0.21",
  doBy = "4.7.1",                 # not loaded
  ftsa = "6.8",
  demography = "2.0.1",           # Lee–Carter, Hyndman–Ullah, coherent FDM (chapter 11)
  pbkrtest = "0.5.5",             # not loaded
  car = "3.1-5",                  # loaded by bayesLife
  bayesLife = "5.3-1"             # UN e0 model (chapter 11 comparator)
)
norm <- function(v) gsub("-", ".", v)

missing <- names(pinned)[!vapply(names(pinned), requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing)          # binaries where available

mismatch <- character(0)
for (pkg in names(pinned)) {
  have <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
  if (is.na(have) || norm(have) != norm(pinned[[pkg]])) mismatch <- c(mismatch, pkg)
  message(sprintf("%-14s installed %-12s paper %s", pkg, have, pinned[[pkg]]))
}
if (length(mismatch)) {
  message("Building pinned versions from source for: ", paste(mismatch, collapse = ", "))
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  for (pkg in mismatch) {
    remotes::install_version(pkg, version = pinned[[pkg]], upgrade = "never", quiet = TRUE)
  }
}
message("Done.")
