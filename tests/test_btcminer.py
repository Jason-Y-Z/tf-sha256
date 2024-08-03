import copy
import random

import numpy as np
import tensorflow as tf

from tests import btcminer
from tf_btcminer import bitcoin, coinbase, constants, encoder, utilities
from tf_sha256 import add32, sha256


def scale(a: tf.Tensor) -> tf.Tensor:
    """Scale a tensor to closer to the [0, 1] boundary."""
    return tf.where(tf.greater(a, 0.5), 1.0, 0.0)


add32.scale_batches = add32.scale_batch = add32.scale = scale


def _bits_to_bin(bits: tf.Tensor) -> str:
    return "".join(map(lambda x: str(int(x)), bits.numpy().tolist()))


def _hex_to_bin(hex_str: str) -> str:
    return bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)


def _bytes_to_bin(bytes_str: bytes) -> str:
    return bin(int.from_bytes(bytes_str, "big"))[2:].zfill(len(bytes_str) * 8)


def test_tf_tx_compute_merkle_root():
    n_seq = random.randint(10, constants.SEQ_LEN)
    cb_hash = tf.where(np.random.uniform(size=256) > 0.5, 1.0, 0.0)
    tx_hashes = tf.where(np.random.uniform(size=(n_seq, 256)) > 0.5, 1.0, 0.0)

    cb_bin = _bits_to_bin(cb_hash)
    cb_hex = ""
    for idx in range(0, 256, 4):
        cb_hex += hex(int(cb_bin[idx : idx + 4], 2))[2:]
    all_hex = [cb_hex]
    txs_hex = []
    for idx in range(tf.shape(tx_hashes)[0]):
        tx_bin = _bits_to_bin(tx_hashes[idx])
        tx_hex = ""
        for idx in range(0, 256, 4):
            tx_hex += hex(int(tx_bin[idx : idx + 4], 2))[2:]
        all_hex.append(tx_hex)
        txs_hex.append(tx_hex)

    merkle_root = btcminer.tx_compute_merkle_root(all_hex)

    tx_parts = encoder.txs_to_parts(txs_hex)
    tf_tx_parts = encoder.txs_to_bits(tx_parts, constants.PARTS_LEN)
    tf_tx_mask = encoder.txs_to_mask(tx_parts, constants.PARTS_LEN)
    tf_merkle_root = coinbase.tf_tx_compute_merkle_root(
        cb_hash, tf_tx_parts, tf_tx_mask
    )

    assert _bits_to_bin(tf_merkle_root) == _hex_to_bin(merkle_root)


def test_block_maker(example_block, address, value):
    n_tx = 10
    example_block["transactions"] = example_block["transactions"][:n_tx]
    example_block["coinbasevalue"] = value
    example_block["coinbase"] = (
        "0400001059124d696e656420627920425443204775696c640800000037".zfill(200)
    )

    block_data = copy.deepcopy(example_block)
    for idx in range(n_tx):
        assert example_block["transactions"][idx]["hash"] == block_data["tx"][idx + 1]
    txs_input = tf.convert_to_tensor(
        encoder.txs_to_bits(block_data["tx"][1 : n_tx + 1]), dtype=tf.float32
    )
    for idx in range(n_tx):
        assert _hex_to_bin(example_block["transactions"][idx]["hash"]).zfill(
            256
        ) == _bits_to_bin(txs_input[idx])

    tx_parts = encoder.txs_to_parts(block_data["tx"][1 : n_tx + 1])
    txs_parts_input = tf.convert_to_tensor(
        encoder.txs_to_bits(tx_parts, constants.PARTS_LEN), dtype=tf.float32
    )
    tx_mask_input = tf.convert_to_tensor(
        encoder.txs_to_mask(tx_parts, constants.PARTS_LEN), dtype=tf.bool
    )

    other_input_df, hb_len = encoder.block_to_bits(block_data)
    other_input = tf.convert_to_tensor(other_input_df, dtype=tf.float32)
    height_bits_len = tf.constant(hb_len * 4)
    pred_enc = tf.convert_to_tensor(
        encoder.block_to_enc_bits(block_data), dtype=tf.float32
    )

    coinbase_script = pred_enc[constants.BIT_WIDTH :]
    assert _bits_to_bin(coinbase_script) == _hex_to_bin(block_data["coinbase"])

    height_bits = other_input[3][-height_bits_len:]
    hec = encoder.tx_encode_coinbase_height(block_data["height"])
    assert _bits_to_bin(height_bits) == _hex_to_bin(hec)

    value = utilities.reverse_bits_as_bytes(
        tf.reshape(other_input[4 + constants.PBH_WIDTH :], (-1,))
    )
    cb_tx = btcminer.tx_make_coinbase(
        block_data["coinbase"],
        address,
        block_data["coinbasevalue"],
        block_data["height"],
    )
    coinbase_tx = coinbase.tf_construct_coinbase(
        value,
        coinbase_script,
        height_bits,
    )
    block_data["tx"][0] = btcminer.tx_compute_hash(cb_tx)
    cb_hash = utilities.reverse_bits_as_bytes(
        sha256.sha256_fixed(sha256.sha256(coinbase_tx))
    )
    assert _bits_to_bin(cb_hash) == _hex_to_bin(block_data["tx"][0])

    merkle_root = btcminer.tx_compute_merkle_root(block_data["tx"][: n_tx + 1])
    tf_tx_parts = encoder.txs_to_bits(tx_parts, constants.PARTS_LEN)
    tf_tx_mask = encoder.txs_to_mask(tx_parts, constants.PARTS_LEN)
    tf_merkle_root = coinbase.tf_tx_compute_merkle_root(
        cb_hash, tf_tx_parts, tf_tx_mask
    )
    assert _bits_to_bin(tf_merkle_root) == _hex_to_bin(merkle_root)

    z_header = bitcoin.tf_make_header(
        txs_parts_input,
        tx_mask_input,
        other_input,
        tf.constant(hb_len * 4),
        pred_enc,
    )
    hash_loss = bitcoin.tf_hash_loss(
        txs_parts_input,
        tx_mask_input,
        other_input,
        tf.constant(hb_len * 4),
        pred_enc,
    )

    header = btcminer.fill_coinbase_and_nonce(
        block_data, address, block_data["coinbase"], block_data["nonce"]
    )[0]

    assert _bytes_to_bin(header) == _bits_to_bin(z_header)
    print(hash_loss)
