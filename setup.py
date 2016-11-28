from setuptools import setup


PACKAGES = [
        'tmnvi',
        'tmnvi.vae',
        'tmnvi.lda',
]

def setup_package():
    setup(
        name="TopicModelsNVI",
        version='0.1.0',
        description='Topic Models built with Neural Variational Inference',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/TopicModelsNVI',
        license='MIT',
        install_requires=['numpy', 'scipy', 'tensorflow'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
