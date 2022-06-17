import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="finite-dipole",
    version="0.1.0",
    author="Tom Vincent",
    author_email="tom.vincent@manchester.ac.uk",
    description="A Python implementation of the finite dipole model for scanning near-field optical microscopy contrast.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],  # TO ADD: LICENSE, OS
    python_requires=">=3.10",
    install_requires=requirements,
)
