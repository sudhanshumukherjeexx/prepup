import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setuptools.setup(
    name="prepup",
    version="0.1.5",
    author="Sudhanshu Mukherjee",
    author_email="sudhanshumukherjeexx@gmail.com",
    description="Prepup is a free, open-source package that lets you open, explore, visualize, and pre-process datasets in your Computer's Terminal.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "prepup = prepup.__init__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
    install_requires=[
        "argparse",
        "polars",
        "termcolor",
        "pyfiglet",
        "blessed",
        "imbalanced_learn",
        "imblearn",
        "joblib",
        "matplotlib",
        "pyarrow",
        "nbclient",
        "nbformat",
        "numpy",
        "pandas",
        "plotext",
        "prefect",
        "prefect_jupyter",
        "pydantic",
        "pyfiglet",
        "pytest",
        "scikit_learn",
        "scipy",
        "seaborn",
        "termcolor",
    ],
)