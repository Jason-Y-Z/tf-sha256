import tensorflow as tf

from tf_btcminer import coinbase, constants, utilities
from tf_sha256 import sha256


@tf.function(
    input_signature=(
        tf.TensorSpec(
            shape=[constants.PARTS_LEN, constants.HASH_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(shape=[constants.PARTS_LEN], dtype=tf.bool),
        tf.TensorSpec(
            shape=[constants.INPUT_HEIGHT, constants.BIT_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(
            shape=[(constants.CB_WIDTH + 1) * constants.BIT_WIDTH], dtype=tf.float32
        ),
    )
)
def tf_make_header(
    txs_input, tx_mask_input, other_input, height_bits_len, pred_enc
) -> tf.Tensor:
    """Create block header given input parameters."""
    nonce = pred_enc[: constants.BIT_WIDTH]
    coinbase_script = pred_enc[constants.BIT_WIDTH :]

    version = utilities.reverse_bits_as_bytes_32(other_input[0])
    curtime = utilities.reverse_bits_as_bytes_32(other_input[1])
    bits = utilities.reverse_bits_as_bytes_32(other_input[2])
    height_bits = other_input[3][-height_bits_len:]
    previousblockhash = utilities.reverse_bits_as_bytes(
        tf.reshape(other_input[4 : 4 + constants.PBH_WIDTH], (-1,))
    )
    value = utilities.reverse_bits_as_bytes(
        tf.reshape(other_input[4 + constants.PBH_WIDTH :], (-1,))
    )
    nonce = utilities.reverse_bits_as_bytes_32(nonce)

    coinbase_tx = coinbase.tf_construct_coinbase(
        value,
        coinbase_script,
        height_bits,
    )
    cb_hash = utilities.reverse_bits_as_bytes_256(
        sha256.sha256_fixed(sha256.sha256(coinbase_tx))
    )
    merkleroot = utilities.reverse_bits_as_bytes_256(
        coinbase.tf_tx_compute_merkle_root(cb_hash, txs_input, tx_mask_input)
    )

    return tf.concat(
        [version, previousblockhash, merkleroot, curtime, bits, nonce], axis=0
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(
            shape=[constants.PARTS_LEN, constants.HASH_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(shape=[constants.PARTS_LEN], dtype=tf.bool),
        tf.TensorSpec(
            shape=[constants.INPUT_HEIGHT, constants.BIT_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(
            shape=[(constants.CB_WIDTH + 1) * constants.BIT_WIDTH], dtype=tf.float32
        ),
    )
)
def tf_hash_loss(
    txs_input, tx_mask_input, other_input, height_bits_len, pred_enc
) -> tf.Tensor:
    """Compute the loss based on the SHA256 hash."""
    blk_header = tf_make_header(
        txs_input,
        tx_mask_input,
        other_input,
        height_bits_len,
        pred_enc,
    )
    blk_hash = utilities.reverse_bits_as_bytes_256(
        sha256.sha256_fixed(sha256.sha256(blk_header))
    )
    base_vals = tf.reverse(tf.range(0, 1, 1 / 256, dtype=tf.float32), axis=(0,))
    loss = tf.reduce_sum(blk_hash * base_vals)
    return loss
