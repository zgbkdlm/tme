from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="tme",
    version="0.1.4",
    author="Zheng Zhao",
    author_email="zz@zabemon.com",
    description="Taylor moment expansion in Python.",
    keywords=["stochastic differential equations",
              "statistics"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL v3 or later",
    url="https://github.com/zgbkdlm/tme",
    download_url="https://github.com/zgbkdlm/tme",
    packages=["tme"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    python_requires='>=3.8',
    install_requires=requirements
)
