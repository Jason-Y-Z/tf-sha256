import tensorflow as tf

from tf_btcminer import constants, utilities
from tf_sha256 import sha256

TX_FRONT = ""
# version
TX_FRONT += "01000000"
# in-counter
TX_FRONT += "01"
# input[0] prev hash
TX_FRONT += "0" * 64
# input[0] prev seqnum
TX_FRONT += "ffffffff"
TX_FRONT_BITS = utilities.hex_to_bits(TX_FRONT)


# input[0] seqnum
TX_BACK = "ffffffff"
# out-counter
TX_BACK += "01"
TX_BACK_BITS = utilities.hex_to_bits(TX_BACK)
ADDRESS = "bc1qmpcdvxj6uumyj6xvzfdu0v5vfgeapdt7x6c8wh"


# Create a pubkey script
# OP_DUP OP_HASH160 <len to push> <pubkey> OP_EQUALVERIFY OP_CHECKSIG
PUBKEY_SCRIPT = (
    "76" + "a9" + "14" + utilities.bitcoinaddress2hash160(ADDRESS) + "88" + "ac"
)
PUBKEY_SCRIPT_BITS = utilities.hex_to_bits(PUBKEY_SCRIPT + "00000000")


@tf.function(
    input_signature=(
        tf.TensorSpec(
            shape=[constants.CB_VAL_WIDTH * constants.BIT_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(
            shape=[constants.CB_WIDTH * constants.BIT_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
    )
)
def tf_construct_coinbase(value, coinbase_script, height_bits) -> tf.Tensor:
    """Update the coinbase transaction with the new extra nonce."""
    coinbase_script = tf.concat((height_bits, coinbase_script), axis=0)

    # input[0] script len
    cb_shape = tf.cast(tf.shape(coinbase_script)[0] // 8, dtype=tf.uint64)
    tx_front = tf.concat([TX_FRONT_BITS, utilities.int2varintbits(cb_shape)], axis=0)

    # input[0] script
    tx_mid = coinbase_script

    # output[0] value
    # output[0] script len
    tx_back = tf.concat(
        [TX_BACK_BITS, value, utilities.int2varintbits(len(PUBKEY_SCRIPT) // 2)],
        axis=0,
    )
    # output[0] script
    # lock-time
    tx_back = tf.concat([tx_back, PUBKEY_SCRIPT_BITS], axis=0)

    return tf.concat((tx_front, tx_mid, tx_back), axis=0)


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[constants.HASH_WIDTH], dtype=tf.float32),
        tf.TensorSpec(
            shape=[constants.PARTS_LEN, constants.HASH_WIDTH], dtype=tf.float32
        ),
        tf.TensorSpec(shape=[constants.PARTS_LEN], dtype=tf.bool),
    )
)
def tf_tx_compute_merkle_root(cb_hash, txs_input, tx_mask_input) -> tf.Tensor:
    """Compute the Merkle Root of a list of transaction parts."""

    # Convert list of ASCII hex transaction hashes into bits
    cb_hash = utilities.reverse_bits_as_bytes_256(cb_hash)
    tx_parts = utilities.reverse_batches_as_bytes(txs_input)

    # Iteratively compute the merkle root hash
    def b(mroot, tx_part):
        concat_tx = tf.concat([mroot, tx_part], axis=0)
        mroot = sha256.sha256_fixed(sha256.sha256(concat_tx))
        return mroot

    merkle_root = tf.foldl(
        b,
        tf.boolean_mask(tx_parts, tx_mask_input),
        cb_hash,
        parallel_iterations=1,
        name="tf_tx_compute_merkle_root_foldl",
    )
    return utilities.reverse_bits_as_bytes_256(merkle_root)
