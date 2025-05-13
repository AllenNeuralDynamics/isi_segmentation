from setuptools import setup, find_packages

VERSION = "0.1.1"

# requirements
install_requires = [
    "scipy==1.12.0",
    "opencv-python==4.9.0.80",
    "gdown==5.1.0",
    "matplotlib==3.8.0",
    "h5py==3.10.0",
    "tensorflow==2.9.0",
]

setup(
    name="isi_segmentation",
    packages=find_packages(),
    version=VERSION,
    description="Supervised ISI segmentaion using tensorflow",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Di Wang",
    install_requires=install_requires,
    author_email="di.wang@alleninstitute.org",
    url="https://github.com/AllenNeuralDynamics/isi_segmentation",
    keywords=["deep learning", "computer vision"],
    python_requires='>=3.7',
)