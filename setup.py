import setuptools

import versioneer

with open("README.rst", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="finite-dipole",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Tom Vincent",
    author_email="tom.vincent@manchester.ac.uk",
    description="A Python implementation of the finite dipole model for scanning near-field optical microscopy contrast.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],  # TO ADD: LICENSE
    python_requires=">=3.9",
    install_requires=requirements,
)
