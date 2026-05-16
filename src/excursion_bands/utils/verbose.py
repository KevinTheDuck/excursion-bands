"""
[utils/verbose.py]
Used for prints methods; pretty straight forward
"""


def logger(tag: str, body: str) -> str:
    return f"{tag: <50} {body}"
