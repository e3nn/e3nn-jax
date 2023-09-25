import nox


@nox.session
def tests(session):
    session.install("pytest")
    session.run("pip", "install", "-e", ".")
    session.install("flax", "dm-haiku", "jraph", "tqdm", "optax")
    session.run("pytest")
