from setuptools import find_packages, setup

setup(
    name="room_env",
    version="0.0.1",
    license="MIT",
    author="Taewoon Kim",
    author_email="tae898@gmail.com",
    packages=find_packages(),
    # packages=find_packages("room_env"),
    # package_dir={"": "room_env"},
    url="https://github.com/tae898/room-env",
    install_requires=[
        "gym>=0.23.1",
        "numpy>=1.22.3",
        "PyYAML>=6.0",
        "requests>=2.27.1",
        "tqdm>=4.64.0",
    ],
    package_data={"": ["data/*"]},
)
