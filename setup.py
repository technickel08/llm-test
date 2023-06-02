import glob
from setuptools import setup, find_packages
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = os.path.join(APP_DIR, 'requirements.txt')

print ("APP_DIR: ", APP_DIR)
print ("REQUIREMENTS_PATH: ", REQUIREMENTS_PATH)


with open(REQUIREMENTS_PATH) as f:
    required = f.read().splitlines()

setup(
    name='bharatpe_ds_api_server',
    version='0.1.0',  # Required
    license='GPL-2.0',
    author="",
    author_email="",
    description="Analytics api server",
    long_description_content_type="text/markdown",
    packages=['ds_api_server'] , # Required
    include_package_data=True,
    install_requires=required,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Flask",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.7",
        "Operating System :: POSIX",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
    ]
)