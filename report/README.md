# Report and article

| File | What it is |
|---|---|
| `rapor_tr.tex` | Turkish course report (IEEE conference format, two columns) |
| `article_en.tex` | Shorter English article on the same work |
| `figures/tr/`, `figures/en/` | Figures (PDF for LaTeX, PNG for slides) and tables (CSV, LaTeX), Turkish and English labels |

**Compile:** upload this folder to Overleaf (New Project → Upload Project, as a zip) and compile `rapor_tr.tex` or `article_en.tex` with pdfLaTeX. `IEEEtran`, `babel` (Turkish) and TikZ are included in Overleaf.

**Rebuild the figures** from the committed result files (no GPU needed):

```
python -c "import json,sys; sys.path.insert(0,'src'); from hdtr import figures as F; L=lambda n: json.load(open(f'results/{n}',encoding='utf-8')); [F.build_all(L('06_train.json'),L('07_thresholds.json'),L('08_final_test.json'),f'report/figures/{g}',native=L('10_native_test.json'),lang=g) for g in ('tr','en')]"
```

Every number in the texts comes from `results/08_final_test.json`, `results/09_exploratory.json`, `results/10_native_test.json` and `results/native_agreement.json`.
