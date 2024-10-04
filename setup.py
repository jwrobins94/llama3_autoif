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
        'flash_attn',
        'lightning',
        'wandb'
    ]
)