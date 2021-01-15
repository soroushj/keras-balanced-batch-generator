import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='keras-balanced-batch-generator',
    version='0.0.1',
    url='https://github.com/soroushj/keras-balanced-batch-generator',
    author='Soroush Javadi',
    author_email='soroush.javadi@gmail.com',
    license='MIT',
    description='A Keras-compatible generator for creating balanced batches',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'keras',
        'generator',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    install_requires=[
        'numpy>=1.0.0',
    ],
    python_requires='>=3.0',
    py_modules=['keras_balanced_batch_generator'],
)
