"""Execute notebooks and export previews.

Default: clean Jupyter kernels. --in-process: separate IPython subprocesses for
restricted environments that cannot bind Jupyter kernel sockets.
"""
from pathlib import Path
import sys
import subprocess
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def execute_without_sockets(path):
    import base64
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from IPython.display import display
    shell = InteractiveShell.instance()

    def show_figures(*args, **kwargs):
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format='png', dpi=110, bbox_inches='tight')
            display({'image/png': base64.b64encode(buffer.getvalue()).decode()}, raw=True)
        plt.close('all')

    plt.show = show_figures
    notebook = nbformat.read(path, as_version=4)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{path.name}, cell {count} failed: '
                               f'{result.error_before_exec or result.error_in_exec}')
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data,
                                                       metadata=output.metadata or {}))
    notebook.metadata['verification'] = {
        'executor': 'IPython, isolated subprocess, no kernel sockets',
        'live_collectors': 'disabled; use refresh flags to validate upstream endpoints'}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    if '--worker' in sys.argv:
        execute_without_sockets(Path(sys.argv[-1]))
        return
    preview = ROOT / 'results/notebook_previews'
    preview.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        if '--in-process' in sys.argv:
            subprocess.run([sys.executable, __file__, '--worker', str(path)], cwd=ROOT, check=True)
            notebook = nbformat.read(path, as_version=4)
        else:
            from nbclient import NotebookClient
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            NotebookClient(notebook, timeout=180, kernel_name='python3',
                           resources={'metadata': {'path': str(ROOT)}}).execute()
            nbformat.write(notebook, path)
        body, _ = HTMLExporter().from_notebook_node(notebook)
        (preview / (path.stem + '.html')).write_text(body)
        print('Executed:', path.name, flush=True)

if __name__ == '__main__':
    main()
