"""Create a clean GitHub directory and a separate portable weights archive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'exports/github_project'
DEST.mkdir(parents=True,exist_ok=True)
FILES = ['research_baseline_and_geometry.ipynb','research_patch_experiments.ipynb','README.md','mesh_quality.ipynb','requirements.txt','pyproject.toml','.gitignore',
         'artifacts/README.md','scripts/export_project.py','scripts/execute_final_notebook.py',
         'tests/test_final_solution.py']
for name in FILES:
    target = DEST/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/name,target)
for name in ['src','configs','reports']:
    shutil.copytree(ROOT/name,DEST/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
archive = ROOT/'exports/final_assets.zip'
with zipfile.ZipFile(archive.with_suffix('.tmp'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
    for path in sorted((ROOT/'artifacts/final').glob('*')):
        if path.is_file():bundle.write(path,path.relative_to(ROOT))
archive.with_suffix('.tmp').replace(archive)
with archive.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
(ROOT/'exports/final_assets.sha256').write_text(f'{checksum}  final_assets.zip\n')
print(f'''Папка для GitHub: {DEST}
Архив весов для Release: {archive}
Размер архива: {archive.stat().st_size / 1024**2:.1f} MiB
''')

code_archive = ROOT / 'exports/github_project.zip'
with zipfile.ZipFile(code_archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(DEST.rglob('*')):
        if path.is_file():
            bundle.write(path, path.relative_to(DEST))
print(f'Архив кода: {code_archive}')
