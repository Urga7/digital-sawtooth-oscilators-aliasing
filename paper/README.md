# Seminar paper (LaTeX)

IEEE conference template (template version 6/27/2024, `IEEEtran.cls` v1.8b) from
`SeminarWorkTemplates26-27`, on A4 paper like `conference-template-a4.docx`.
(`Seminal work templates` holds the older Elektrotehniški vestnik journal style and an
older `IEEEtran.cls` v1.7a from 2007; it isn't used here.)

## Files

| File | Content | Suggested length |
|---|---|---|
| `main.tex` | title, author, structure; nothing to write here except title/author | |
| `sections/00-abstract.tex` | abstract + keywords | ~100 words |
| `sections/01-introduction.tex` | I. Introduction | ~200 words |
| `sections/02-overview.tex` | II. Overview of the three references | ~400 words |
| `sections/03-contribution.tex` | III. Your experiment: oscillators, method, results (figures + table), discussion | ~550 words |
| `sections/04-conclusions.tex` | IV. Conclusions | ~150 words |
| `sections/05-appendix-critical-observations.tex` | Appendix – Critical Observations (after Conclusions, before References) | 300 words ±5 % |
| `refs.bib` | references (only cited ones are printed) | |

All sections are written. The text merges this structure with the content of the earlier
draft in `C:\Git\digital-sawtooth-oscilators-aliasing\paper\seminar.tex`, rewritten for
readers new to audio and updated to the current results. `\fillin{...}` (red text) is
still defined for new placeholders; none are left. The appendix is a draft written from
the documented session events (see the comment at the top of its file) and should be
checked against the session PDFs.

Style rules for the running text: no long dashes, colons or semicolons.

Figures are read directly from `../experiment/figures/`, so re-running the Python
scripts updates the paper too. Figs. 1 and 2 come from `paper_figures.py`, Figs. 3 to 5
from `run_experiment.py`. Every number in the text is taken from
`../experiment/results/` (`summary.json`, `cpu.csv`, `demo_aliasing.csv`).

## Build

```
latexmk -pdf main.tex
```

or `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`.
On Overleaf, upload this folder and the `experiment/figures` folder, and change
`\graphicspath` in `main.tex` accordingly.

## Word count

```
python wordcount.py
```

prints words per section and checks them against the limits (paper 1330–1470,
appendix 285–315). It counts running text, headings and captions separately and leaves
out math, citations, table cells, the bibliography and unfilled `\fillin` placeholders.
Whether headings, captions and the abstract count towards the 1400 words is not stated
in the instructions, so it prints all variants; ask if in doubt. The appendix is
excluded from Overleaf's word count by `%TC:ignore` markers.
