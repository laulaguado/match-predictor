import nbformat

with open('notebooks/01_preparacion_datos.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

cell = nb.cells[12]
print('Original:')
print(cell.source[:200])

# Fix
new_src = cell.source.replace('"%% len(nc)"', '"% len(nc)"')
new_src = new_src.replace('print("Variables numericas: %d\\n", %% len(nc))', 'print("Variables numericas: %d" %% len(nc))')
new_src = new_src.replace('"%% %%"', '"%%"')
cell.source = new_src

print('\nFixed:')
print(cell.source[:200])

with open('notebooks/01_preparacion_datos.ipynb', 'w') as f:
    nbformat.write(nb, f)

print('\nSaved')
