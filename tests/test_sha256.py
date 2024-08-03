import hashlib

import numpy as np
import tensorflow as tf

from tests import sha256
from tf_sha256 import add32
from tf_sha256 import sha256 as tf_sha256_


@tf.function()
def scale(a: tf.Tensor) -> tf.Tensor:
    """Scale a tensor to closer to the [0, 1] boundary."""
    return tf.where(tf.greater(a, 0.5), 1.0, 0.0)


add32.scale_batches = add32.scale_batch = add32.scale = scale


def _bits_to_bin(bits: tf.Tensor) -> str:
    return "".join(map(lambda x: str(int(x)), bits.numpy().tolist()))


def test_add32():
    a = tf.where(np.random.uniform(size=32) > 0.5, 1.0, 0.0)
    b = tf.where(np.random.uniform(size=32) > 0.5, 1.0, 0.0)
    a_int = int(_bits_to_bin(a), 2)
    b_int = int(_bits_to_bin(b), 2)
    c = add32.add32_2(a, b)
    c_bin = _bits_to_bin(c)
    assert sha256.add32(a_int, b_int) == int(c_bin, 2)


def test_add32_batch_mode():
    a = tf.where(np.random.uniform(size=(10, 32)) > 0.5, 1.0, 0.0)
    b = tf.where(np.random.uniform(size=(10, 32)) > 0.5, 1.0, 0.0)
    c = add32.add32_2_batch(a, b)
    for idx in range(tf.shape(a)[0]):
        a_int = int(_bits_to_bin(a[idx]), 2)
        b_int = int(_bits_to_bin(b[idx]), 2)
        c_bin = _bits_to_bin(c[idx])
        assert sha256.add32(a_int, b_int) == int(c_bin, 2)


def test_message_schedule_array():
    block = tf.where(np.random.uniform(size=512) > 0.5, 1.0, 0.0)
    block_bin = _bits_to_bin(block)
    block_bytes = b""
    for idx in range(0, 512, 8):
        block_bytes += int(block_bin[idx : idx + 8], 2).to_bytes(
            length=1, byteorder="big"
        )
    ws = tf.unstack(tf_sha256_.message_schedule_array(tf.stack(block, axis=0)), axis=0)
    w_ints = sha256.message_schedule_array(block_bytes)

    for w, w_int in zip(ws, w_ints):
        assert w_int == int(_bits_to_bin(w), 2)


def test_message_schedule_array_batch_mode():
    block = tf.where(np.random.uniform(size=(10, 512)) > 0.5, 1.0, 0.0)
    ws = tf.unstack(
        tf_sha256_.message_schedule_array_batch(tf.stack(block, axis=0)), axis=0
    )

    for block_idx in range(tf.shape(block)[0]):
        ws_slice = [w[block_idx] for w in ws]
        block_bin = _bits_to_bin(block[block_idx])
        block_bytes = b""
        for idx in range(0, 512, 8):
            block_bytes += int(block_bin[idx : idx + 8], 2).to_bytes(
                length=1, byteorder="big"
            )
        w_ints = sha256.message_schedule_array(block_bytes)

        for w, w_int in zip(ws_slice, w_ints):
            assert bin(w_int) == bin(int(_bits_to_bin(w), 2))


def test_round():
    state = tf_sha256_.IV_TENSORS
    round_constant = tf_sha256_.ROUND_TENSORS[0]
    round_const_int = sha256.ROUND_CONSTANTS[0]
    assert int(_bits_to_bin(round_constant), 2) == round_const_int

    schedule_word = tf.where(np.random.uniform(size=32) > 0.5, 1.0, 0.0)
    round_tensors = tf_sha256_.round_(state, round_constant, schedule_word)
    w_int = int(_bits_to_bin(schedule_word), 2)
    round_ints = sha256.round_(sha256.IV, round_const_int, w_int)

    for round_int, round_tensor in zip(round_ints, round_tensors):
        assert round_int == int(_bits_to_bin(round_tensor), 2)


def test_round_batch_mode():
    n_batches = 10
    state = []
    for tensor in tf_sha256_.IV_TENSORS:
        state.append(tf_sha256_.tile_batch(tensor, n_batches))
    round_constant = tf_sha256_.tile_batch(tf_sha256_.ROUND_TENSORS[0], n_batches)
    round_const_int = sha256.ROUND_CONSTANTS[0]
    schedule_word = tf.where(np.random.uniform(size=(n_batches, 32)) > 0.5, 1.0, 0.0)
    round_tensors = tf_sha256_.round_batch(state, round_constant, schedule_word)

    for idx in range(n_batches):
        w_int = int(_bits_to_bin(schedule_word[idx]), 2)
        round_ints = sha256.round_(sha256.IV, round_const_int, w_int)
        round_tensor_slices = [t[idx] for t in round_tensors]

        for round_int, round_tensor in zip(round_ints, round_tensor_slices):
            assert round_int == int(_bits_to_bin(round_tensor), 2)


def test_sha256():
    input_bin = "000000010000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111111111111111111111111111111101101000000000110101100111011111000011000000001101011001110111110000110000011011010011010110100101101110011001010110010000100000011000100111100100100000010000010110111001110100010100000110111101101111011011000011100000110111001101001111111000000000000000000000000000101011010011011000001111111000111110101011111001101101011011011010001001110101001001100000010000010001101011001111010100110000101010010011101101101111100110011011100011101010011110001100010101111110101000111101101001011001101001001101110010101001000001101100111101100100111110110101010000110001100111001111011110000001000100000000000000000000000000000000000000000000000000000000000010100100000010010000000000000000010000010101001000000011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111000000011010000001100011010111001001010100000000000000000000000000000000000110010111011010101001000101000010011110100001111100010010011101110001110111100101110011000011101101110011100101000001011001100100101100100101001101111100000101010011000101101011111001000011100010001010110000000000000000000000000000000000"
    input_tensor = tf.constant(list(map(int, input_bin)), dtype=tf.float32)
    bit_hash = tf_sha256_.sha256(input_tensor)

    input_bytes = b""
    for idx in range(0, len(input_bin), 8):
        input_bytes += int(input_bin[idx : idx + 8], 2).to_bytes(
            length=1, byteorder="big"
        )
    normal_hash = sha256.sha256(input_bytes)[0]
    lib_hash = hashlib.sha256(input_bytes).digest()

    assert int.from_bytes(normal_hash, byteorder="big") == int(
        _bits_to_bin(bit_hash), 2
    )
    assert int.from_bytes(lib_hash, byteorder="big") == int.from_bytes(
        normal_hash, byteorder="big"
    )


def test_sha256_batch_mode():
    n_batches = 12
    input_len = 120
    input_tensor = tf.where(
        np.random.uniform(size=(n_batches, input_len)) > 0.5, 1.0, 0.0
    )
    bit_hash = tf_sha256_.sha256_batch(input_tensor)

    for batch_idx in range(n_batches):
        input_bin = "".join(
            map(lambda x: str(int(x)), input_tensor[batch_idx].numpy().tolist())
        )
        input_bytes = b""
        for idx in range(0, len(input_bin), 8):
            input_bytes += int(input_bin[idx : idx + 8], 2).to_bytes(
                length=1, byteorder="big"
            )
        normal_hash = sha256.sha256(input_bytes)[0]
        lib_hash = hashlib.sha256(input_bytes).digest()

        assert int.from_bytes(normal_hash, byteorder="big") == int(
            _bits_to_bin(bit_hash[batch_idx]), 2
        )
        assert int.from_bytes(lib_hash, byteorder="big") == int.from_bytes(
            normal_hash, byteorder="big"
        )
