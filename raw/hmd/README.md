# The HMD data go here

Save the Human Mortality Database's bulk file of all statistics of the release of
**29 October 2026** here, unopened and not renamed:

    raw/hmd/hmd_statistics_20261029.zip

Get it from <https://www.mortality.org> after registering (free): its DOI,
[10.4054/HMD.Countries.20261029](https://doi.org/10.4054/HMD.Countries.20261029), leads to that release, which is also in the
*Previous Versions* table of the *Data → Zipped data files* page once a newer release is out.
Then check it from the repository folder:

    uv run mdmx data verify

The HMD asks its users to obtain the data from its website and not to pass them on, so nothing in
this folder is ever committed (`.gitignore`).
