# MDMx: the computational companion of the arXiv article

This repository is the computational companion of

> Clark, S. J. (2026). *Multi-dimensional Mortality (MDMx): One Tensor Decomposition for Model Life
> Tables, Fitting, Summary-Indicator Prediction, and Forecasting.* arXiv:2603.20518
> ([2603.20518v3](https://arxiv.org/abs/2603.20518v3)).

It reproduces every figure, table and number in the article from the Human Mortality Database (HMD),
and explains each step in more detail than the article can.  You can read it without running anything:
the rendered companion is at **<https://sinafala.github.io/mdmx-arxiv/arxiv-v3/>**.

This copy is the snapshot **`arxiv-v3`**, which accompanies version v3 of the article.
Each version of the article on arXiv has a snapshot of its own (a git tag of this repository and a
folder of the rendered book), so the links in each version of the article lead to the code and the
results as they were for that version.  [`SNAPSHOTS.md`](SNAPSHOTS.md) lists them.

## Contents

* [What the companion is](#what-the-companion-is)
* [How the article links here](#how-the-article-links-here)
* [What is in this repository](#what-is-in-this-repository)
* [Reproducing the results](#reproducing-the-results)
  1. [Install the tools](#1-install-the-tools)
  2. [Get this repository](#2-get-this-repository)
  3. [Set up the Python and R environments](#3-set-up-the-python-and-r-environments)
  4. [Get the HMD data](#4-get-the-hmd-data)
  5. [Render the companion](#5-render-the-companion)
  6. [Run bayesLife for chapter 12](#6-run-bayeslife-for-chapter-12)
  7. [Check that the results reproduce](#7-check-that-the-results-reproduce)
* [Working with the companion](#working-with-the-companion)
* [Troubleshooting](#troubleshooting)
* [Citing and licence](#citing-and-licence)

## What the companion is

MDMx treats the single-age life tables of the HMD as one four-way array of the logit of the
probability of dying, by sex, age, population and year, and decomposes it into sex and age bases
that every population and year share.  On that basis the article builds a typology of mortality age
patterns, schedules at any level of mortality, a model of wars and epidemics, a life table fitter,
predictions from child and adult mortality, and two forecasters, which it evaluates against
established methods from rolling forecast origins.

The companion is a [Quarto](https://quarto.org) book with one chapter for each part of that work.
Each chapter runs its code when the book is rendered, shows it next to the prose that explains it,
and writes what the article uses:

* figures to `companion/results/figures/` and tables to `companion/results/tables/`;
* every number the article quotes, as a LaTeX macro, to `companion/results/numbers/` (and all of them
  to `companion/results/numbers.json`, with a description and the cell that computed it);
* a record of every one of these files, with the chapter that made it and the checksums of its
  inputs, to `companion/results/manifest.json`, and every change made to the published data to
  `companion/results/ledger/data-changes.csv`.

The article reads only these files, so the article and the companion cannot disagree: no number in
the article is typed by hand.  The routine machinery lives in the Python package `mdmx`
(`src/mdmx/`), whose functions say in their docstrings which equation of the article they implement.

| Chapter (file) | What it does | Article |
|---|---|---|
| `data/d1-acquire` – `d6-tensor` | Verifies the HMD release, reads it, curates it, labels the years of wars and pandemics, pools sparse years, and builds the tensor | §2 |
| `analysis/04-tucker` | The shared sex and age bases and the effective cores | §3 |
| `analysis/06-clustering` | The typology of mortality age patterns | §4 |
| `analysis/07-trajectories` | Schedules at any level of mortality | §5 |
| `analysis/08-exceptional` | Exceptional mortality: baselines, disruption profiles and intensities | §6 |
| `analysis/09-fitting` | The life table fitter and its evaluation | §7 |
| `analysis/10-svdcomp` | Prediction of schedules from summary indicators | §8 |
| `analysis/11-forecasting` | The two forecasters, and what the flow field says about mortality change | §9, §11.4 |
| `analysis/12-evaluation` | The forecast evaluation | §10 |
| `appendices/` | The computing environment and the provenance of every result | §12 |

The chapters keep their own file numbers, which do not follow the article's sections; the table, and
the box at the top of each chapter, give the correspondence.

## How the article links here

The article is written so that every result can be traced to the code that made it.  In the PDF:

* each equation and each method carries a mark **code**, which opens the function that implements it
  in this repository, at the lines it occupies in snapshot `arxiv-v3`
  (`https://github.com/sinafala/mdmx-arxiv/blob/arxiv-v3/src/mdmx/…#L…`);
* each figure and table carries the marks **book** and **source**: the cell of the rendered companion
  that made it, and that cell's lines in the chapter's source here;
* each number is a link to the cell of the rendered companion that computed it.

Because the links name the snapshot, they keep pointing at the right lines when later versions of the
companion change the code.

## What is in this repository

| Path | What it holds |
|---|---|
| `companion/` | The Quarto book: `index.qmd`, `data/` (Part I, the data pipeline), `analysis/` (Part II, one chapter per part of the article), `appendices/`, and its configuration `_quarto.yml` |
| `companion/results/` | The figures, tables, numbers, ledger and manifest that the article reads, as this snapshot's render wrote them |
| `src/mdmx/` | The Python package with all the computation; the chapters only call it |
| `config/` | Every tunable value (`params.toml`, `params.d/`), the event dictionary of wars and pandemics (`events.csv`), the population register (`populations.csv`), and the headline results on the HMD's previous release (`release-20260218.json`) |
| `R/` | The R scripts of the comparison methods (Lee–Carter, the functional-data models, bayesLife with MortCast), and `install-packages.R`, which installs the R packages at the versions that ran |
| `raw/` | `MANIFEST.json`, the checksums of the HMD release; you put the downloaded data in `raw/hmd/` |
| `tools/` | The few command-line tools the chapters name: bayesLife's off-render runs (`bayeslife_bundle.py`, `fetch_bayeslife_offsite.py`), runs computed ahead of a render (`warm_evaluation.py`, `warm_variants.py`), the record of the previous release (`freeze_previous_release.py`), and `compare_numbers.py` |
| `tests/` | The unit tests that the chapters cite (the network, the summary-indicator models, the forecasters) |
| `pyproject.toml`, `uv.lock` | The Python project and the exact versions of every Python package |
| `LICENSE.md`, `LICENSES/`, `NOTICE` | The licences |

Not here: the HMD data, which the HMD's terms ask every user to download from its website; the
outputs of bayesLife's runs, which are derived from those data (step 6 makes them); and the cache of
intermediate results, which a render rebuilds.

## Reproducing the results

Everything below works on macOS, Windows and Linux, and every tool is free.  The versions in brackets
are those that produced this snapshot; newer versions of Quarto and R normally work, while the Python
packages are pinned exactly by `uv.lock` and the R packages by `R/install-packages.R`.

What it takes: about 2 GB of disk for the tools, 7 GB for the cache of intermediate results, and time.
On an Apple M1 Max (10 cores) a first render of the whole companion takes about 3½ hours, most of it
the comparison methods of chapter 12; bayesLife's 48 runs for chapter 12 take about 12 hours of
computing more (step 6), and can be spread over several machines.  Rendering again afterwards takes
minutes, because finished steps are cached.

### 1. Install the tools

| Tool | What it is for |
|---|---|
| **Git** | getting this repository |
| **uv** | installs Python (3.14) and the exact package versions |
| **Quarto** (1.9) | renders the book and runs its code |
| **R** (4.5) | runs the comparison methods of chapters 11 and 12 |
| An **HMD account** | the data (free) |

**Git.**  macOS: `xcode-select --install` in Terminal (or `brew install git`).  Windows: install *Git
for Windows* from <https://git-scm.com/download/win>, then run once `git config --global
core.longpaths true`.  Linux: your package manager.

**uv.**  macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`.  Windows (PowerShell):
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.  Open a new
terminal and check `uv --version`.  uv installs Python itself, so you need no other Python.

**Quarto.**  Download the installer for your system from <https://quarto.org/docs/get-started/>,
then check with `quarto check` in a new terminal.

**R.**  Install R from <https://cloud.r-project.org>.  On Windows install **Rtools** too (same site),
and on macOS Apple's command-line tools (`xcode-select --install`): an R package whose pinned version
is older than CRAN's is built from source, which needs them.  If `Rscript` is not on your `PATH`, set
the environment variable `MDMX_RSCRIPT` to its full path.

### 2. Get this repository

```bash
git clone --branch arxiv-v3 https://github.com/sinafala/mdmx-arxiv.git
cd mdmx-arxiv
```

`--branch arxiv-v3` checks out this snapshot.  Keep the folder out of cloud-synced folders (iCloud
Drive, OneDrive, Dropbox), whose sync clients fight with the thousands of small files of the Python
environment and the cache.

### 3. Set up the Python and R environments

From the repository folder:

```bash
uv sync --frozen --extra dev   # .venv with exactly the versions in uv.lock
Rscript R/install-packages.R   # the R packages at their pinned versions (once; takes a while)
```

The `dev` extra adds the test tools and PyYAML, which Quarto's Jupyter engine needs.

### 4. Get the HMD data

1. Register (free) at <https://www.mortality.org>.
2. Download the bulk file of all statistics of the release of **29 October 2026**, DOI
   [10.4054/HMD.Countries.20261029](https://doi.org/10.4054/HMD.Countries.20261029) (the file `hmd_statistics_20261029.zip`).  The
   DOI leads to that release; once a newer release is out, it is also in the *Previous Versions*
   table of the HMD's *Data → Zipped data files* page.
3. Save it, unopened and not renamed, as `raw/hmd/hmd_statistics_20261029.zip` in this
   repository.
4. Check it:

   ```bash
   uv run mdmx data verify
   ```

   This recomputes the file's SHA-256 checksums and compares them with `raw/MANIFEST.json`.  Every
   result of the article comes from this release; another release gives other numbers.

The HMD asks its users to obtain the data from its website and not to pass them on, so neither this
repository nor anything it makes publishes them; keep your copy, and bayesLife's runs (step 6), on
your own computer.

### 5. Render the companion

```bash
uv run mdmx render --project companion
```

This runs every chapter in order, writes `companion/results/` and the book to
`companion/_output/` (open `companion/_output/index.html` in a browser).  Always pass
`--project companion`: the command serves more than one book in the author's development
repository.  To render one chapter, name its file:

```bash
uv run mdmx render --project companion 06-clustering
```

A chapter reads the cached results of the chapters before it; if they are missing, it stops and says
which chapter to render first.  The first render stops at chapter 12, because bayesLife's runs are
not there yet: chapters d1–11 are complete at that point, and step 6 makes the runs.

### 6. Run bayesLife for chapter 12

Chapter 12 compares the forecasters with bayesLife, the United Nations' model of life expectancy (the
R package `bayesLife`).  Its 48 runs take about 12 hours of computing on a recent Mac, so no render
runs them: they run as a separate R project, which chapter 12 then reads.

```bash
# 1. chapter 12 once more (if step 5 did not just do it): it computes bayesLife's inputs, then stops
uv run mdmx render --project companion 12-evaluation
# 2. write the R project from those inputs
MDMX_PROJECT=companion uv run python tools/bayeslife_bundle.py --out work/bayeslife/offsite-batches/<date>_hmd-20261029
# 3. run it (R only; its README.md says how to share the runs between machines and follow progress)
cd work/bayeslife/offsite-batches/<date>_hmd-20261029 && bash run.sh
# 4. back in the repository folder: put the runs where chapter 12 reads them, then render it
MDMX_PROJECT=companion uv run python tools/fetch_bayeslife_offsite.py --source work/bayeslife/offsite-batches/<date>_hmd-20261029
uv run mdmx render --project companion 12-evaluation
```

On Windows PowerShell set the variable first (`$env:MDMX_PROJECT = "companion"`) and drop it from the
command; run step 3 in Git Bash.  Chapter 12 accepts a run only if its inputs and settings are
exactly those the companion makes, so runs made from other data or settings are refused, not used.

### 7. Check that the results reproduce

Your render rewrote `companion/results/`.  Compare its numbers with the snapshot's:

```bash
git show arxiv-v3:companion/results/numbers.json > numbers-arxiv-v3.json
uv run python tools/compare_numbers.py numbers-arxiv-v3.json companion/results/numbers.json
```

The script lists every number that differs by more than 0.1 %.  Expect a few: run times differ from
machine to machine, and linear algebra libraries (Apple's Accelerate, OpenBLAS, MKL) differ in their
last digits, which optimisers that stop at a tolerance can carry a little further.  The unit tests the
chapters cite run with `uv run pytest tests`.

## Working with the companion

**Caching.**  Every expensive step is stored in `work/companion/artifacts/` with a record of its
inputs, its parameters, the code it ran (every `mdmx` function it reaches), the Python lock file and,
for steps that run R, the versions of all R packages.  A step is recomputed only when one of these
changes, so a render after a change recomputes what the change affects and nothing else.  To
recompute everything, set `MDMX_FORCE=1` (macOS and Linux: `MDMX_FORCE=1 uv run mdmx render --project
companion`; PowerShell: `$env:MDMX_FORCE = "1"` first).  `uv run mdmx clean --project companion
--work` deletes the cache and the rendered book.

**Changing a setting.**  Every tunable value is in `config/params.toml` or `config/params.d/`.  Change
it and render the chapter again: only the steps that use it recompute, and the numbers in
`companion/results/` change with them.

**Interactive work.**  Open the folder in an editor that runs Quarto documents (Positron, VS Code with
the Quarto extension, RStudio) with the `.venv` interpreter, open a chapter and run its cells in order;
the first cell of every chapter sets everything up, and every chapter loads its inputs from the cache.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `quarto: command not found` | Open a new terminal after installing Quarto; check `quarto check`. |
| `Kernel not found` when rendering | Render through uv (`uv run mdmx render …`), which puts `.venv` first. |
| `HMD release … not found` | The zip must be exactly `raw/hmd/hmd_statistics_20261029.zip`. |
| `SHA-256 … ≠ manifest` | The file is another HMD release, or was opened and saved again: download it again. |
| An R step fails to load a package | Run `Rscript R/install-packages.R` again; it reports each package's version against its pin. |
| Chapter 12 stops, asking for bayesLife's runs | Step 6. |
| Every step recomputes after an update of R or of a package | Expected: the cache's keys hold the versions, and a step recomputes when its versions change. |

## Citing and licence

Please cite the article, and the Human Mortality Database as its terms ask:

> Clark, S. J. (2026). *Multi-dimensional Mortality (MDMx).* arXiv:2603.20518.
>
> HMD. Human Mortality Database. Max Planck Institute for Demographic Research (Germany), University of
> California, Berkeley (USA), and French Institute for Demographic Studies (France). Available at
> www.mortality.org (data downloaded on [date]), DOI: 10.4054/HMD.Countries.20261029.

[`CITATION.cff`](CITATION.cff) has the same in machine-readable form (GitHub's *Cite this repository*).

The code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md) and the text and
results under [CC BY-NC 4.0](LICENSES/); [`NOTICE`](NOTICE) gives the required notices.  Commercial use
needs a separate licence from the author.
