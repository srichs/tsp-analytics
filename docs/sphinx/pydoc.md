# pydoc output

`pydoc` is a lightweight way to generate HTML directly from the docstrings in
`tsp.tsp`. It is a useful companion to Sphinx when you want a quick, single-file
reference output.

## Generate HTML

```bash
python -m pydoc -w tsp.tsp
```

The command writes `tsp.tsp.html` in the current working directory. You can move
that file into `docs/sphinx/_build/html` or any other static hosting location.
