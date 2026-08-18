# TECH 65000

Book-style course site built with Jupyter Book and deployed with GitHub Pages.

## Local preview

```bash
python -m pip install -r requirements.txt
jupyter-book build .
python -m http.server --directory _build/html
```
