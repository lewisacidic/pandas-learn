from invoke import run, task
import os

@task
def linter():
    run('pylint pdlearn')

@task
def tests(pytest=False):
    if pytest:
        run('py.test')
    else:
        run('nosetests')

@task
def clean(bytecode=False, docs=False, checkpoints=False):
    if checkpoints:
        run(r'find . | grep -E ".ipynb_checkpoints" | xargs rm -r')
    if bytecode:
        run(r'find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -r')
    if docs:
        run('rm -r docs/build')

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
