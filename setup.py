from setuptools import setup, find_packages

print(find_packages())
setup(
    name="al_ilqr",
    packages=find_packages(),
      install_requires=[
          'numpy', 'torch', 'matplotlib', 'pytorch3d'
      ],

)