from invoke import run, task
import webbrowser
import os

@task
def linter():
    run('pylint pdlearn')

@task
def tests(html=False):
    opts = ['py.test']
    if html:
        opts.append('--cov-report=html')
    run(' '.join(opts))
    if html:
        webbrowser.open('file://' + os.path.realpath('htmlcov/index.html'))

@task
def clean(all=False, bytecode=False, docs=False,
          checkpoints=False, coverage=False):
    if checkpoints or all:
        run(r'find . | grep -E ".ipynb_checkpoints" | xargs rm -r')
    if bytecode or all:
        run(r'find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -r')
    if docs or all:
        run('rm -r docs/build')
    if coverage or all:
        run('rm -r htmlcov')

@task
def docs(interactive=False, api=False, notebooks=False):
    if api:
        run('sphinx-apidoc -o docs/source/api pdlearn')
    if notebooks:
        for root, sub, files in os.walk('docs'):
            for f in files:
                if os.path.splitext(f)[1] == '.ipynb':
                    run("""
                    cd {root}
                    jupyter nbconvert {file} --to rst
                    """.format(root=root, file=f))
    if interactive:
        run('sphinx-autobuild --open-browser docs/source docs/build/html')
    else:
        run('sphinx-build docs/source docs/build/html')
