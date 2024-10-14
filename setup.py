from setuptools import setup, find_packages
from os import path


setup(name = 'PyDeePC',
    packages=find_packages(),
    version = '0.3.6',
    description = 'Python library for Data-Enabled Predictive Control',
    url = 'https://github.com/rssalessio/PyDeePC',
    author = 'Alessio Russo',
    author_email = 'arusso2@bu.edu',
    install_requires=['numpy', 'scipy', 'cvxpy', 'matplotlib', 'jupyter', 'ecos'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)