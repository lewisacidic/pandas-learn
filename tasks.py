from invoke import run, task

@task
def lint():
    run("pylint pdlearn")

@task
def test(pytest=False):
    if pytest:
        run("py.test")
    else:
        run("nosetests")

