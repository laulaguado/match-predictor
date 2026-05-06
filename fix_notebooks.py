# Fix notebooks - remove special characters
import nbformat
import re

def clean_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for cell in nb.cells:
        if hasattr(cell, 'source'):
            # Replace checkmarks and special chars
            cell.source = cell.source.replace('✓', '[OK]')
            cell.source = cell.source.replace('✗', '[NO]')
            cell.source = cell.source.replace('💕', '(heart)')
            cell.source = cell.source.replace('✅', '[OK]')
            cell.source = cell.source.replace('📊', '[Chart]')
            cell.source = cell.source.replace('🎯', '[Target]')
            cell.source = cell.source.replace('🏠', '[Home]')
            cell.source = cell.source.replace('📋', '[Doc]')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Cleaned: {filepath}')

clean_notebook('notebooks/01_preparacion_datos.ipynb')
clean_notebook('notebooks/02_modelamiento.ipynb')
print('Done')
