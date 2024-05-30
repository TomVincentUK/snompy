import setuptools

import versioneer

with open("README.rst", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="snompy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Tom Vincent",
    author_email="TomVincentCode@gmail.com",
    description=(
        "A Python package for modelling scanning near-field optical "
        "microscopy measurements."
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
