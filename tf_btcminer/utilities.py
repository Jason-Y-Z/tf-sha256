import numpy as np
import tensorflow as tf

from tf_btcminer import constants
from tf_sha256 import sha256


def bitcoinaddress2hash160(addr):
    """
    Convert a Base58 Bitcoin address to its Hash-160 ASCII hex string.

    Args:
        addr (string): Base58 Bitcoin address

    Returns:
        string: Hash-160 ASCII hex string
    """

    table = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    hash160 = 0
    addr = addr[::-1]
    for i, c in enumerate(addr):
        hash160 += (58**i) * table.find(c)

    # Convert number to 50-byte ASCII Hex string
    hash160 = f"{hash160:050x}"

    # Discard 1-byte network byte at beginning and 4-byte checksum at the end
    return hash160[2 : 50 - 8]


def hex_to_bits(hex_str: str) -> tf.Tensor:
    num = int(hex_str, 16)
    bits = np.zeros(len(hex_str) * 4)
    for idx, _ in enumerate(bits):
        bits[idx] = num % 2
        num //= 2
    bits = bits[::-1]
    return tf.constant(bits, dtype=tf.float32)


@tf.function(
    input_signature=(tf.TensorSpec(shape=[constants.BIT_WIDTH], dtype=tf.float32),)
)
def reverse_bits_as_bytes_32(bits) -> tf.Tensor:
    rev_bytes = tf.reverse(tf.reshape(bits, (-1, 8)), axis=(0,))
    return tf.reshape(rev_bytes, (-1,))


@tf.function(
    input_signature=(tf.TensorSpec(shape=[constants.HASH_WIDTH], dtype=tf.float32),)
)
def reverse_bits_as_bytes_256(bits) -> tf.Tensor:
    rev_bytes = tf.reverse(tf.reshape(bits, (-1, 8)), axis=(0,))
    return tf.reshape(rev_bytes, (-1,))


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
def reverse_bits_as_bytes(bits) -> tf.Tensor:
    rev_bytes = tf.reverse(tf.reshape(bits, (-1, 8)), axis=(0,))
    return tf.reshape(rev_bytes, (-1,))


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, sha256.HASH_WIDTH], dtype=tf.float32),)
)
def reverse_batches_as_bytes(batches) -> tf.Tensor:
    n_batches = tf.shape(batches)[0]
    rev_bytes = tf.reverse(tf.reshape(batches, (n_batches, -1, 8)), axis=(1,))
    return tf.reshape(rev_bytes, (n_batches, -1))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[], dtype=tf.uint64),
        tf.TensorSpec(shape=[], dtype=tf.uint64),
    )
)
def _int2lebits(value, width) -> tf.Tensor:
    return reverse_bits_as_bytes(sha256.int2bebits(value, width))


_FD_BITS = hex_to_bits("fd")
_FE_BITS = hex_to_bits("fe")
_FF_BITS = hex_to_bits("ff")


@tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.uint64),))
def int2varintbits(value) -> tf.Tensor:
    if tf.less(value, 0xFD):
        return _int2lebits(value, 1)
    elif tf.less_equal(value, 0xFFFF):
        return tf.concat([_FD_BITS, _int2lebits(value, 2)], axis=0)
    elif tf.less_equal(value, 0xFFFFFFFF):
        return tf.concat([_FE_BITS, _int2lebits(value, 4)], axis=0)
    else:
        return tf.concat([_FF_BITS, _int2lebits(value, 8)], axis=0)
