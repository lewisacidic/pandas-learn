from invoke import run, task

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
def clean(bytecode=False, docs=False):
    if bytecode:
        run('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
    if docs:
        run('rm -r docs/build')
@task
def docs(build=False):
    if build:
        run('sphinx-build docs/source docs/build/html')
    else:
        run('sphinx-autobuild --open-browser docs/source docs/build/html')
