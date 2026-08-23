"""
Quantum number utility functions.
"""

__all__ = ["encode_quantum_number_pair", "decode_quantum_number_pair"]


def encode_quantum_number_pair(qa: int, qb: int):
    """
    Encode a pair of quantum numbers into a single quantum number.
    """
    return (qa << 16) + qb


def decode_quantum_number_pair(qnum: int):
    """
    Decode a quantum number into two separate quantum numbers.
    """
    qb = qnum % (1 << 16)
    if qb >= (1 << 15):
        qb -= (1 << 16)
    elif qb < -(1 << 15):
        qb += (1 << 16)
    qa = (qnum - qb) >> 16
    return qa, qb
