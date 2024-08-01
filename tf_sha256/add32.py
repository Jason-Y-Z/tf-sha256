"""Overloads of Fixed-length Additions for tf Graphs."""

import keras
import tensorflow as tf
from tf_sha256 import constants


BIT_WIDTH = 32
_NO_CARRY = tf.where(tf.greater(tf.range(BIT_WIDTH), 0), 1.0, 0.0)


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),))
def scale(a: tf.Tensor) -> tf.Tensor:
    """Scale a tensor to closer to the [0, 1] boundary."""
    mean: float = 0.5
    std: float = 3.0
    return keras.activations.sigmoid((a - mean) * std)


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),)
)
def scale_batch(a: tf.Tensor) -> tf.Tensor:
    """Scale a tensor to closer to the [0, 1] boundary."""
    mean: float = 0.5
    std: float = 3.0
    return keras.activations.sigmoid((a - mean) * std)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), None, BIT_WIDTH], dtype=tf.float32),
    )
)
def scale_batches(a: tf.Tensor) -> tf.Tensor:
    """Scale a tensor to closer to the [0, 1] boundary."""
    mean: float = 0.5
    std: float = 3.0
    return keras.activations.sigmoid((a - mean) * std)


@tf.function(input_signature=(tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),))
def _carry(sum_: tf.Tensor) -> tf.Tensor:
    init_carry = tf.ones_like(sum_, dtype=tf.float32)

    def _more_to_carry(_, carry: tf.Tensor) -> bool:
        return tf.reduce_sum(carry) > 0

    def _carry(sum_: tf.Tensor, _) -> tuple[tf.Tensor, tf.Tensor]:
        """Look at the sum tensor and carry 1 in binary."""
        carry_mask = tf.greater_equal(sum_, 1.5)
        carry_deduction = tf.where(carry_mask, -2.0, 0.0)
        carry = tf.where(carry_mask, 1.0, 0.0)
        carry = tf.roll(carry * _NO_CARRY, -1, axis=tf.constant(0))
        sum_ += carry_deduction + carry
        return sum_, carry

    return tf.while_loop(_more_to_carry, _carry, (sum_, init_carry))[0]


@tf.function(
    input_signature=(tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),)
)
def _carry_1(sum_: tf.Tensor) -> tf.Tensor:
    init_carry = tf.ones_like(sum_, dtype=tf.float32)

    def _more_to_carry(_, carry: tf.Tensor) -> bool:
        return tf.reduce_sum(carry) > 0

    def _carry(sum_: tf.Tensor, _) -> tuple[tf.Tensor, tf.Tensor]:
        """Look at the sum tensor and carry 1 in binary."""
        carry_mask = tf.greater_equal(sum_, 1.5)
        carry_deduction = tf.where(carry_mask, -2.0, 0.0)
        carry = tf.where(carry_mask, 1.0, 0.0)
        carry = tf.roll(carry * _NO_CARRY, -1, axis=tf.constant(1))
        sum_ += carry_deduction + carry
        return sum_, carry

    return tf.while_loop(_more_to_carry, _carry, (sum_, init_carry))[0]


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), BIT_WIDTH], dtype=tf.float32),
    )
)
def _carry_1_iv(sum_: tf.Tensor) -> tf.Tensor:
    init_carry = tf.ones_like(sum_, dtype=tf.float32)

    def _more_to_carry(_, carry: tf.Tensor) -> bool:
        return tf.reduce_sum(carry) > 0

    def _carry(sum_: tf.Tensor, _) -> tuple[tf.Tensor, tf.Tensor]:
        """Look at the sum tensor and carry 1 in binary."""
        carry_mask = tf.greater_equal(sum_, 1.5)
        carry_deduction = tf.where(carry_mask, -2.0, 0.0)
        carry = tf.where(carry_mask, 1.0, 0.0)
        carry = tf.roll(carry * _NO_CARRY, -1, axis=tf.constant(1))
        sum_ += carry_deduction + carry
        return sum_, carry

    return tf.while_loop(_more_to_carry, _carry, (sum_, init_carry))[0]


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), None, BIT_WIDTH], dtype=tf.float32),
    )
)
def _carry_2_iv(sum_: tf.Tensor) -> tf.Tensor:
    init_carry = tf.ones_like(sum_, dtype=tf.float32)

    def _more_to_carry(_, carry: tf.Tensor) -> bool:
        return tf.reduce_sum(carry) > 0

    def _carry(sum_: tf.Tensor, _) -> tuple[tf.Tensor, tf.Tensor]:
        """Look at the sum tensor and carry 1 in binary."""
        carry_mask = tf.greater_equal(sum_, 1.5)
        carry_deduction = tf.where(carry_mask, -2.0, 0.0)
        carry = tf.where(carry_mask, 1.0, 0.0)
        carry = tf.roll(carry * _NO_CARRY, -1, axis=tf.constant(2))
        sum_ += carry_deduction + carry
        return sum_, carry

    return tf.while_loop(_more_to_carry, _carry, (sum_, init_carry))[0]


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_2(a, b):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b))
    sum_ = _carry(sum_)
    return scale(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_2_batch(a, b):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b))
    sum_ = _carry_1(sum_)
    return scale_batch(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[len(constants.IV), BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_2_batch_iv(a, b):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b))
    sum_ = _carry_1_iv(sum_)
    return scale_batch(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[len(constants.IV), None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[len(constants.IV), None, BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_2_batches_iv(a, b):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b))
    sum_ = _carry_2_iv(sum_)
    return scale_batches(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_4(a, b, c, d):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b, c, d))
    sum_ = _carry(sum_)
    return scale(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_4_batch(a, b, c, d):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b, c, d))
    sum_ = _carry_1(sum_)
    return scale_batch(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_5(a, b, c, d, e):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b, c, d, e))
    sum_ = _carry(sum_)
    return scale(sum_)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
        tf.TensorSpec(shape=[None, BIT_WIDTH], dtype=tf.float32),
    )
)
def add32_5_batch(a, b, c, d, e):
    """Sum the args but within width 32."""
    sum_ = tf.math.add_n((a, b, c, d, e))
    sum_ = _carry_1(sum_)
    return scale_batch(sum_)
