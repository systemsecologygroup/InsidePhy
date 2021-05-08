import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="insidephy",
    version="0.0.1b13",
    author="Esteban Acevedo-Trejos",
    author_email="esteban.acevedo@leibniz-zmt.de",
    description="insidephy: A package for modelling inter- and intraspecific size variability of phytoplankton",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SEGGroup/",
    packages=setuptools.find_packages(include=['insidephy', 'insidephy.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Education",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.7',
    include_package_data=True,
    install_requires=['numpy>=1.19.2', 'scipy>=1.5.2', 'matplotlib>=3.3.2', 'pandas>=1.1.3', 'seaborn>=0.11.0',
                      'xarray>=0.16.1', 'dask>=2.30.0', 'tables>=3.6.1'],
)
