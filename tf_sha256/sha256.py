"""Bit-based SHA256 Hash Algorithm."""

import logging
from typing import Sequence

import numpy as np
import tensorflow as tf

from tf_sha256 import add32, constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

BIT_WIDTH = 32
HASH_WIDTH = 256


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    )
)
def tile_batch(vec, batch_size) -> tf.Tensor:
    """Tile the vector with a batch dimension."""
    vec = tf.expand_dims(vec, axis=0)
    vec = tf.tile(vec, [batch_size, 1])
    return vec


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    )
)
def rightrotate32(x, n):
    """Right rotate at width 32."""
    return tf.roll(x, n, axis=tf.constant(0))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    )
)
def rightrotate32_batch(x, n):
    """Right rotate at width 32."""
    return tf.roll(x, n, axis=tf.constant(1))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    ),
)
def bitwise_or(a, b) -> tf.Tensor:
    """Bitwise OR for binary tensors."""
    return add32.scale(a + b)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    ),
)
def bitwise_or_batch(a, b) -> tf.Tensor:
    """Bitwise OR for binary tensors."""
    return add32.scale_batch(a + b)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    ),
)
def bitwise_and(a, b) -> tf.Tensor:
    """Bitwise AND for binary tensors."""
    return add32.scale(a * b)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    ),
)
def bitwise_and_batch(a, b) -> tf.Tensor:
    """Bitwise AND for binary tensors."""
    return add32.scale_batch(a * b)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    ),
)
def bitwise_xor(a, b) -> tf.Tensor:
    """Bitwise XOR for binary tensors."""
    xor = bitwise_or(a, b) - bitwise_and(a, b)
    return add32.scale(xor)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    ),
)
def bitwise_xor_batch(a, b) -> tf.Tensor:
    """Bitwise XOR for binary tensors."""
    xor = bitwise_or_batch(a, b) - bitwise_and_batch(a, b)
    return add32.scale_batch(xor)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    )
)
def right_shift(a, n) -> tf.Tensor:
    """Bitwise right shift by n bits on a."""
    rm_mask = tf.where(tf.greater(tf.range(BIT_WIDTH), n - 1), 1.0, 0.0)
    rolled_a = tf.roll(a, n, axis=tf.constant(0))
    return rolled_a * rm_mask


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    )
)
def right_shift_batch(a, n) -> tf.Tensor:
    """Bitwise right shift by n bits on a."""
    rm_mask = tf.where(tf.greater(tf.range(BIT_WIDTH), n - 1), 1.0, 0.0)
    rm_mask = tile_batch(rm_mask, tf.shape(a)[0])
    rolled_a = tf.roll(a, n, axis=tf.constant(1))
    return rolled_a * rm_mask


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),))
def little_sigma0(word) -> tf.Tensor:
    """Little sigma 0 formula."""
    return bitwise_xor(
        bitwise_xor(rightrotate32(word, 7), rightrotate32(word, 18)),
        right_shift(word, 3),
    )


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),)
)
def little_sigma0_batch(word) -> tf.Tensor:
    """Little sigma 0 formula."""
    return bitwise_xor_batch(
        bitwise_xor_batch(rightrotate32_batch(word, 7), rightrotate32_batch(word, 18)),
        right_shift_batch(word, 3),
    )


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),))
def little_sigma1(word) -> tf.Tensor:
    """Little sigma 1 formula."""
    return bitwise_xor(
        bitwise_xor(rightrotate32(word, 17), rightrotate32(word, 19)),
        right_shift(word, 10),
    )


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),)
)
def little_sigma1_batch(word) -> tf.Tensor:
    """Little sigma 1 formula."""
    return bitwise_xor_batch(
        bitwise_xor_batch(rightrotate32_batch(word, 17), rightrotate32_batch(word, 19)),
        right_shift_batch(word, 10),
    )


ROUND_SIZE = 64


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH * 16], dtype=tf.float32),))
def message_schedule_array(block) -> tf.Tensor:
    """Compute the message schedule array."""
    w = tf.reshape(block, shape=(16, BIT_WIDTH))

    def c(idx, w_):
        return tf.less(idx, tf.shape(w_)[0])

    def b(idx, w_):
        s0 = little_sigma0(w_[-15])
        s1 = little_sigma1(w_[-2])
        w_i_new = add32.add32_4(w_[-16], s0, w_[-7], s1)
        w_i_new = tf.expand_dims(w_i_new, axis=0)
        return idx + 1, tf.concat([w_[1:], w_i_new], axis=0)

    w_2 = tf.while_loop(
        c, b, [tf.constant(0), w], name="message_schedule_array_foldl_first_half"
    )[1]
    w = tf.concat([w, w_2], axis=0)
    w_3 = tf.while_loop(
        c, b, [tf.constant(0), w], name="message_schedule_array_foldl_second_half"
    )[1]
    return tf.concat([w, w_3], axis=0)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    )
)
def _partial_msg_schedule_batch(w, i):
    s0 = little_sigma0_batch(w[i - 15])
    s1 = little_sigma1_batch(w[i - 2])
    return add32.add32_4_batch(w[i - 16], s0, w[i - 7], s1)


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH * 16], dtype=tf.float32),)
)
def message_schedule_array_batch(block) -> tf.Tensor:
    """Compute the message schedule array."""
    w = tf.transpose(
        tf.reshape(block, shape=(tf.shape(block)[0], 16, BIT_WIDTH)), perm=[1, 0, 2]
    )

    def c(idx, _):
        return tf.less(idx, ROUND_SIZE)

    def b(idx, w):
        w1 = tf.expand_dims(_partial_msg_schedule_batch(w, idx), axis=0)
        w2 = tf.expand_dims(_partial_msg_schedule_batch(w, idx + 1), axis=0)
        return idx + 2, tf.concat([w, w1, w2], axis=0)

    return tf.while_loop(
        c,
        b,
        [tf.constant(16), w],
        shape_invariants=[
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None, BIT_WIDTH], dtype=tf.float32),
        ],
        parallel_iterations=1,
        name="message_schedule_array_batch_while_loop",
    )[1]


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),))
def big_sigma0(word) -> tf.Tensor:
    """Big sigma 0 array."""
    return bitwise_xor(
        bitwise_xor(rightrotate32(word, 2), rightrotate32(word, 13)),
        rightrotate32(word, 22),
    )


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),)
)
def big_sigma0_batch(word) -> tf.Tensor:
    """Big sigma 0 array."""
    return bitwise_xor_batch(
        bitwise_xor_batch(rightrotate32_batch(word, 2), rightrotate32_batch(word, 13)),
        rightrotate32_batch(word, 22),
    )


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),))
def big_sigma1(word) -> tf.Tensor:
    """Big sigma 1 array."""
    return bitwise_xor(
        bitwise_xor(rightrotate32(word, 6), rightrotate32(word, 11)),
        rightrotate32(word, 25),
    )


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),)
)
def big_sigma1_batch(word) -> tf.Tensor:
    """Big sigma 1 array."""
    return bitwise_xor_batch(
        bitwise_xor_batch(rightrotate32_batch(word, 6), rightrotate32_batch(word, 11)),
        rightrotate32_batch(word, 25),
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    ),
)
def choice(x, y, z) -> tf.Tensor:
    """Choice between y and z with x."""
    return bitwise_xor(bitwise_and(x, y), bitwise_and(1 - x, z))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    ),
)
def choice_batch(x, y, z) -> tf.Tensor:
    """Choice between y and z with x."""
    return bitwise_xor_batch(bitwise_and_batch(x, y), bitwise_and_batch(1 - x, z))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    ),
)
def majority(x, y, z) -> tf.Tensor:
    """Majority among x, y and z."""
    return bitwise_xor(
        bitwise_xor(bitwise_and(x, y), bitwise_and(x, z)), bitwise_and(y, z)
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    ),
)
def majority_batch(x, y, z) -> tf.Tensor:
    """Majority among x, y and z."""
    return bitwise_xor_batch(
        bitwise_xor_batch(bitwise_and_batch(x, y), bitwise_and_batch(x, z)),
        bitwise_and_batch(y, z),
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    )
)
def round_(
    state,
    round_constant,
    schedule_word,
) -> tf.Tensor:
    """Round state given the constant and schedule word."""
    s1 = big_sigma1(state[4])
    ch = choice(state[4], state[5], state[6])
    temp1 = add32.add32_5(state[7], s1, ch, round_constant, schedule_word)
    s0 = big_sigma0(state[0])
    maj = majority(state[0], state[1], state[2])
    temp2 = add32.add32_2(s0, maj)
    return tf.stack(
        [
            add32.add32_2(temp1, temp2),
            state[0],
            state[1],
            state[2],
            add32.add32_2(state[3], temp1),
            state[4],
            state[5],
            state[6],
        ],
        axis=0,
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    )
)
def round_batch(
    state,
    round_constant,
    schedule_word,
) -> tf.Tensor:
    """Round state given the constant and schedule word."""
    s1 = big_sigma1_batch(state[4])
    ch = choice_batch(state[4], state[5], state[6])
    temp1 = add32.add32_5_batch(state[7], s1, ch, round_constant, schedule_word)
    s0 = big_sigma0_batch(state[0])
    maj = majority_batch(state[0], state[1], state[2])
    temp2 = add32.add32_2_batch(s0, maj)
    return tf.stack(
        [
            add32.add32_2_batch(temp1, temp2),
            state[0],
            state[1],
            state[2],
            add32.add32_2_batch(state[3], temp1),
            state[4],
            state[5],
            state[6],
        ],
        axis=0,
    )


def _int_constants_to_tensors(constants_: Sequence[int], name: str) -> tf.Tensor:
    tensors = []
    for c in constants_:
        c_array = np.zeros((BIT_WIDTH,))
        for idx in range(BIT_WIDTH):
            c_array[idx] = c % 2
            c //= 2
        c_array = c_array[::-1]
        c_tensor = tf.constant(c_array, dtype=tf.float32)
        tensors.append(c_tensor)
    tensors = np.vstack(tensors)
    return tf.constant(tensors, dtype=tf.float32, name=name)


ROUND_TENSORS = _int_constants_to_tensors(constants.ROUND_CONSTANTS, "ROUND_CONSTANTS")
IV_TENSORS = _int_constants_to_tensors(constants.IV, "IV")


@tf.function
def _compress_step(state_words, round_params):
    round_constant, schedule_word = round_params
    return round_(state_words, round_constant, schedule_word)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[64 * 8], dtype=tf.float32),
    )
)
def _compress_block(input_state_words, block) -> tf.Tensor:
    """Compress an input block."""
    w = message_schedule_array(block)

    state_words = tf.foldl(
        _compress_step,
        (ROUND_TENSORS, w),
        input_state_words,
        name="compress_block_foldl",
    )
    return add32.add32_2_batch_iv(input_state_words, state_words)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 64 * 8], dtype=tf.float32),
    )
)
def _compress_block_batch(input_state_words, block) -> tf.Tensor:
    """Compress a input block."""
    w = message_schedule_array_batch(block)

    def b(state_words, round_params):
        round_constant, schedule_word = round_params
        round_constant = tile_batch(round_constant, tf.shape(block)[0])
        return round_batch(state_words, round_constant, schedule_word)

    state_words = tf.foldl(
        b,
        (ROUND_TENSORS, w),
        input_state_words,
        name="compress_block_batch_foldl",
    )
    return add32.add32_2_batches_iv(input_state_words, state_words)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[], dtype=tf.uint64),
        tf.TensorSpec(shape=[], dtype=tf.uint64),
    )
)
def int2bebits(value, width) -> tf.Tensor:
    """Integer to big-endian bits."""
    shifts = tf.range(width * 8, dtype=tf.int64)
    shifts = tf.cast(shifts, dtype=tf.uint64)
    shifts = tf.where(tf.less_equal(shifts, BIT_WIDTH * 2), shifts, BIT_WIDTH * 2)
    shifts = shifts[::-1]
    factors = tf.bitwise.left_shift(tf.ones_like(shifts, dtype=tf.uint64), shifts)
    num = tf.cast(value, dtype=tf.uint64)
    bits = tf.cast((tf.bitwise.bitwise_and(num, factors)), dtype=tf.float64) / tf.cast(
        factors, dtype=tf.float64
    )
    bits = tf.cast(bits, dtype=tf.float32)
    return bits


@tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),))
def padding_bytes(input_len) -> tf.Tensor:
    """Pad the input to a specific length."""
    remainder_bytes = (input_len + 8) % 64
    filler_bytes = 64 - remainder_bytes
    zero_bytes = filler_bytes - 1
    input_bits_len = tf.cast(8 * input_len, dtype=tf.uint64)
    encoded_bit_length = int2bebits(input_bits_len, 8)
    prefix = tf.constant([1] + [0] * 7, dtype=tf.float32)
    zeros = tf.repeat(tf.zeros(8, dtype=tf.float32), (zero_bytes,))
    padding = tf.concat([prefix, zeros, encoded_bit_length], axis=0)
    return padding


@tf.function
def _compress_block_step(state_words_, block):
    state_words_ = _compress_block(state_words_, block)
    return state_words_


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
def sha256(message) -> tf.Tensor:
    """SHA256 hash for bit tensors."""
    msg_len = tf.shape(message)[0]
    padding = padding_bytes(msg_len // 8)
    padded = tf.concat([message, padding], axis=0)
    padded = tf.reshape(padded, (-1, 64 * 8))

    state_words = tf.foldl(
        _compress_block_step,
        padded,
        IV_TENSORS,
        parallel_iterations=1,
        name="sha256_foldl",
    )
    return tf.concat(tf.unstack(state_words, axis=0), axis=0)


@tf.function(input_signature=(tf.TensorSpec(shape=[HASH_WIDTH], dtype=tf.float32),))
def sha256_fixed(message) -> tf.Tensor:
    """SHA256 hash for bit tensors."""
    padding = padding_bytes(HASH_WIDTH // 8)
    padded = tf.concat([message, padding], axis=0)
    padded = tf.reshape(padded, (-1, 64 * 8))

    state_words = tf.foldl(
        _compress_block_step,
        padded,
        IV_TENSORS,
        parallel_iterations=1,
        name="sha256_fixed_foldl",
    )
    return tf.concat(tf.unstack(state_words, axis=0), axis=0)


@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
def sha256_batch(message) -> tf.Tensor:
    """SHA256 hash for bit tensors."""
    msg_len = tf.shape(message)[1]
    padding = padding_bytes(msg_len // 8)
    padding = tile_batch(padding, tf.shape(message)[0])
    padded = tf.concat([message, padding], axis=1)

    def c(i, _):
        return tf.less(i, tf.shape(padded)[1])

    def b(i, state_words):
        block = padded[:, i : i + 64 * 8]
        state_words = _compress_block_batch(state_words, block)
        i += 64 * 8
        return i, state_words

    init_words = IV_TENSORS
    init_words = tf.expand_dims(init_words, axis=1)
    init_words = tf.tile(init_words, [1, tf.shape(message)[0], 1])
    state_words = tf.while_loop(
        c,
        b,
        [tf.constant(0), init_words],
        parallel_iterations=1,
        name="sha256_batch_while_loop",
    )[1]
    return tf.concat(tf.unstack(state_words, axis=0), axis=1)
