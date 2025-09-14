"""
Setup script for TextLens package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="textlens",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive text analysis bot for linguistic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/textlens",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/textlens/issues",
        "Documentation": "https://github.com/yourusername/textlens#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "advanced": [
            "spacy>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "wordcloud>=1.9.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "textlens=analyzer:main",
        ],
    },
)
