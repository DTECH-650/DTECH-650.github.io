# TECH 65000

Book-style course site built with Jupyter Book and deployed with GitHub Pages.

## Local preview

```bash
git switch dev
./.venv/bin/jupyter-book clean . --html
./.venv/bin/jupyter-book build .
python -m http.server 8000 --bind 127.0.0.1 --directory _build/html
```

Then open:

http://127.0.0.1:8000/
