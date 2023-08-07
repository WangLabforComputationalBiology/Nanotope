from setuptools import setup, find_packages

setup(
    name='Nanotop',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'biopython==1.79',
        'antiberty',
        'torch',
        'torch_geometric==2.3.1',
        'torch_cluster==1.6.0+pt112cu113',
        'torch-scatter==2.0.9',
        'torch-sparse==0.6.15+pt112cu113',
        'torch-spline-conv==1.2.1+pt112cu113'
    ],
    author='Mengxiangpeng',
    author_email='202002020117@ stumail.sztu.edu.com',
    description='Nanobody paratope prediction using EGNN',
    url='https://github.com/Wo-oh-oh-ooh-oh/Nanotope',
)
