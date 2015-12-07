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
def clean():
    run("""find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf""")
