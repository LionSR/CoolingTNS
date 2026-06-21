# GaussianPaper Build Artifacts

The tracked TeX sources `journal_main.tex` and `journal_supp.tex` cite the
tracked bibliography file `library.bib`.  The bibliography entries in that file
correspond to the citation keys used by the draft.

The publication figure PDFs are not tracked in this repository.  The local
`preamble.tex` deliberately renders labelled placeholder boxes when those PDFs
are absent, so draft builds remain reproducible from tracked files.  A
publication-ready release should either add the final figure PDFs, add scripts
that regenerate them, or document the external artifact location.

The currently referenced figure files are:

- `Fig1.pdf`
- `Fig2.pdf`
- `Fig2inset.pdf`
- `Fig3.pdf`
- `Fig3inset2.pdf`
- `Fig4smallg.pdf`
- `Fig4compareg.pdf`
- `Fig5.pdf`
- `Fig5inset2.pdf`
- `Fig6.pdf`
- `Fig7.pdf`
- `Fig8.pdf`
- `NewFig`
- `NewFig2`
- `Fig9`
- `NewFig3_combined.pdf`
- `Fig2opt.pdf`
- `Fig2opt_inset.pdf`
- `Fig13_cooling_theta-specific.pdf`
- `Fig14_dsp_theta-specific.pdf`
- `Fig15_cooling_phase-averaged.pdf`
- `Fig16_dsp_phase-averaged.pdf`
- `Fig18_g01_energy_spectrum_g01_spec.pdf`
- `newFig17_contour_cooling.pdf`
- `newFig20_contour_DSP.pdf`
- `FigActual18_dsp_phase-averaged.pdf`
- `FigActual19_cooling_phase-averaged.pdf`
- `FigActual21_cooling_scalability.pdf`
