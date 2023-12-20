import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Solar-Panel-Detection-using-Sentinel-2"
AUTHOR_USER_NAME = "Nayal17"
SRC_REPO = "SolarPanelDetection"
AUTHOR_EMAIL = "nayal17.916@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python package for solar panel detection using satellite(Sentinel-2) images",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"}, # ""-> relative location to setup.py
    packages=setuptools.find_packages(where="src")
)
