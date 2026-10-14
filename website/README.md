# Project page

A static page with no build step: `index.html` plus the images and data under `static/`.
It is published to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`
that touches this folder (enable Pages in the repository settings with **GitHub Actions**
as the source).

Preview locally through a web server, since the page loads its data with `fetch`:

```bash
python -m http.server --directory website 8000    # then open http://localhost:8000
```

The example images and their data (`static/rounds*`, `static/gallery*`) are exported from
the paper's editing run by `python tools/build_website_assets.py`. The figures in
`static/figs/` are the paper figures from `analysis/paper_figs.py`, resized for the web.

Before launch, replace the three placeholder links in `index.html` (arXiv paper, Hugging Face and X thread)
and the arXiv id in the BibTeX block.
