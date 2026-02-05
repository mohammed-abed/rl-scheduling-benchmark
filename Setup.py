from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl-scheduling-benchmark",
    version="0.1.0",
    author="Mohammed Aabed ",
    author_email="maabed90@students.iugaza.edu.ps",
    description="Reinforcement Learning for Dynamic Task Scheduling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohammed-abed/rl-scheduling-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0", "black>=22.0.0"],
    },
)

