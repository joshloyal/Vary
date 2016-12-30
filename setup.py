from setuptools import setup


PACKAGES = [
        'vary',
        'vary.variational_autoencoder',
        'vary.information_bottlekneck'
]

def setup_package():
    setup(
        name="Vary",
        version='0.1.0',
        description='Neural Variational Inference Algorithms',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/Vary',
        license='MIT',
        install_requires=['numpy', 'scipy', 'tensorflow'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
