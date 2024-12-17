from setuptools import find_packages, setup

setup(
    name="conformal-safety-learning",
    version="0.1",
    packages=find_packages(where="Scripts"),
    package_dir={"": "Scripts"},
    python_requires=">=3.8,<3.9",  # Ensure Python version 3.8
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "polytope",
        "cvxpy",
        "open3d",
        "control",
        "opencv-python",
        "albumentations"
    ]
)