from setuptools import setup, find_packages


with open("README.md", encoding="utf-8") as f:
    _long_description = f.read()


setup(
    name="e3nn-jax",
    version="0.20.2",
    description="Equivariant convolutional neural networks for the group E(3).",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    author="e3nn team",
    author_email="geiger.mario@gmail.com",
    packages=find_packages(exclude=["examples", "misc", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "jax",
        "jaxlib",
        "sympy",
        "numpy",
        "attrs",
    ],
    extras_require={
        "dev": [
            "plotly",
            "kaleido",
            "jraph",
            "flax",
            "dm-haiku",
            "optax",
            "tqdm",
            "pytest",
            "nox",
        ],
    },
    url="https://e3nn-jax.readthedocs.io",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
