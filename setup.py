from setuptools import setup, find_packages

setup(
    name="df-rnn",
    author="Stellina Ao",
    author_email="stellina@ucla.edu",
    description="Fitting various RNNs to corticostriatal circuit activity during a dynamic foraging task.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    url="https://github.com/stellinaao/df-rnns",
    version="0.0.1" 
)