"""MDMx — Multi-dimensional Mortality.

The ``mdmx`` package holds every reusable piece of the analysis behind the
paper (arXiv:2603.20518) so that the book's chapters can stay short and
readable.  Modules:

``paths``       where files live (project-root relative; works on any OS)
``config``      analysis parameters from ``config/params.toml``
``artifacts``   saving/loading/caching results without pickles, with provenance
``provenance``  run logs and the results manifest
``latex``       tables and in-text numbers written for the paper
``plotting``    figure style, colours, saving figures for the paper
``lifetable``   life-table arithmetic (one implementation used everywhere)
``data``        HMD ingest, curation, events, pooling, tensor (book Part I)
``tensor``      tensor algebra and the decomposition (chapter 04)
``smoothing``   smoothing of the age basis (chapter 04)
"""

__version__ = "3.0.0.dev0"
