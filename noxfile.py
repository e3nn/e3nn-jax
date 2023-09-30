import nox


@nox.session
def tests(session):
    session.install("pytest")
    session.run("pip", "install", ".")
    session.run("python", "examples/irreps_array.py")
    session.run("pip", "install", ".[dev]")
    session.run("pytest")
