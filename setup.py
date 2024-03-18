from setuptools import setup

VERSION = "0.1.0"

# requirements
install_requires = [
    "scipy",
    "opencv-python",
    "gdown",
    "matplotlib",
    "tensorflow",
]

setup(
    name="isi_segmentation",
    packages=["isi_segmentation"],
    version=VERSION,
    description="Supervised ISI segmentaion using tensorflow",
    long_description=open('README.md').read(),
    author="Di Wang",
    install_requires=install_requires,
    author_email="di.wang@alleninstitute.org",
    url="https://github.com/AllenNeuralDynamics/isi_segmentation",
    keywords=["deep learning", "computer vision"],
)