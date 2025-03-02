from setuptools import setup, find_packages

setup(
    name="VectorXGBoost",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "xgboost",
        "numpy",
        "scikit-learn"
    ],
    author="Your Name",
    description="An enhanced XGBoost classifier that supports jagged array features and multi-layer learning.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)