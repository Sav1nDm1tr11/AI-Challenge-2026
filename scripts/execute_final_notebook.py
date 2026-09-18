"""Execute inference/reporting cells; RUN_TRAINING remains False."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path(__file__).resolve().parents[1]
path = root/'mesh_quality.ipynb'
notebook = nbformat.read(path,as_version=4)
client = NotebookClient(notebook,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(root)}})
try:
    client.execute()
finally:
    # Keep static tables and figures; drop transient progress-bar widget handles.
    notebook.metadata.pop('widgets', None)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = [out for out in cell.outputs
                            if 'application/vnd.jupyter.widget-view+json' not in out.get('data', {})]
    nbformat.write(notebook,path)
print('Final notebook executed; training disabled.')
