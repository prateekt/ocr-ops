import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="ocr-ops",
    version="0.0.0.1",
    author="Prateek Tandon",
    author_email="prateek1.tandon@gmail.com",
    description="OCR-Ops Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prateekt/ocr-ops",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.7",
    install_requires=required,
)
