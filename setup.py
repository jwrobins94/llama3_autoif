from setuptools import setup, find_packages

setup(
    name='self_play_ifeval',
    version='0.1',
    packages=find_packages(),
    install_requires=[ # TODO
        'transformers',
        'torch',
        'lm_eval',
        'langdetect',
        'immutabledict',
        'deepspeed',
        'lightning',
        'wandb',
        'accelerate',
        'jinja2>=3.1.0',
        'huggingface_hub',

        'ipykernel',
        'numpy'
    ]
)