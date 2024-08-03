import collections
import hashlib
import random
from typing import Sequence

import jsonlines
import numpy as np
import tensorflow as tf

from tf_btcminer import constants


def fill_digits(
    num: int, matrix: np.ndarray, row_idx: int, base: int = constants.HEX_BASE
):
    """Fill the digits in the matrix."""
    idx = matrix.shape[1] - 1
    while num > 0:
        digit = num % base
        matrix[row_idx][idx] = digit
        num //= base
        idx -= 1


def txs_to_bits(txs: Sequence[str], n_txs: int = constants.SEQ_LEN) -> np.ndarray:
    """Convert hex-formatted transactions to bits."""
    seq = np.zeros((n_txs, constants.HASH_WIDTH), dtype=np.float32)
    for idx, tx in enumerate(txs):
        tx_num = int(tx, 16)
        for tx_idx in range(constants.HASH_WIDTH):
            hex_idx = constants.HASH_WIDTH - tx_idx - 1
            seq[idx][hex_idx] = tx_num % 2
            tx_num //= 2
    return seq


def txs_to_parts(txs: Sequence[str]) -> Sequence[str]:
    """Convert transactions to the parts to be used for merkle root computation."""
    parts = []
    txs_deque = collections.deque()
    for tx in txs:
        txs_deque.append(bytes.fromhex(tx)[::-1])

    while len(txs_deque) > 1:
        new_txs = collections.deque()
        parts.append(txs_deque.popleft()[::-1].hex())
        if len(txs_deque) % 2:
            txs_deque.append(txs_deque[-1])
        while len(txs_deque):
            # Concatenate the next two
            concat = txs_deque.popleft() + txs_deque.popleft()
            # Hash them
            concat_hash = hashlib.sha256(hashlib.sha256(concat).digest()).digest()
            # Add them to our working list
            new_txs.append(concat_hash)
        txs_deque = new_txs
    parts.append(txs_deque[0][::-1].hex())
    return parts


def txs_to_mask(txs: Sequence[str], n_txs: int = constants.SEQ_LEN) -> np.ndarray:
    mask = [True] * len(txs) + [False] * (n_txs - len(txs))
    mask = np.array(mask)
    return mask


def tx_encode_coinbase_height(height, return_len: bool = False):
    """
    Encode the coinbase height, as per BIP 34:
    https://github.com/bitcoin/bips/blob/master/bip-0034.mediawiki

    Arguments:
        height (int): height of the mined block

    Returns:
        string: encoded height as an ASCII hex string
    """

    width = (height.bit_length() + 7) // 8

    height_enc = bytes([width]).hex() + height.to_bytes(width, byteorder="little").hex()
    if return_len:
        return height_enc, len(height_enc)
    else:
        return height_enc


def block_to_bits(block_data_dict: dict) -> tuple[np.ndarray, int]:
    """Convert a block data to a bits array."""
    seq = np.zeros((constants.INPUT_HEIGHT, constants.BIT_WIDTH), dtype=np.float32)
    fill_digits(block_data_dict["version"], seq, 0, constants.BIT_BASE)
    fill_digits(block_data_dict["curtime"], seq, 1, constants.BIT_BASE)
    fill_digits(int(block_data_dict["bits"], 16), seq, 2, constants.BIT_BASE)
    height, hb_len = tx_encode_coinbase_height(
        block_data_dict["height"], return_len=True
    )
    fill_digits(int(height, 16), seq, 3, constants.BIT_BASE)

    pbh = block_data_dict["previousblockhash"]
    for idx in range(constants.PBH_WIDTH):
        seq_str = pbh[idx * constants.HEX_WIDTH : (idx + 1) * constants.HEX_WIDTH]
        if seq_str:
            seq_num = int(seq_str, 16)
            fill_digits(seq_num, seq, idx + 4, constants.BIT_BASE)

    cb_val = hex(block_data_dict["coinbasevalue"])[2:].zfill(
        constants.CB_VAL_WIDTH * constants.HEX_WIDTH
    )
    for idx in range(constants.CB_VAL_WIDTH):
        seq_str = cb_val[idx * constants.HEX_WIDTH : (idx + 1) * constants.HEX_WIDTH]
        if seq_str:
            seq_num = int(seq_str, 16)
            fill_digits(seq_num, seq, idx + 4 + constants.PBH_WIDTH, constants.BIT_BASE)

    return seq, hb_len


def block_to_enc_bits(block_data_dict: dict) -> np.ndarray:
    nonce_enc = np.zeros((1, constants.BIT_WIDTH), dtype=np.float32)
    fill_digits(block_data_dict["nonce"], nonce_enc, 0, constants.BIT_BASE)

    cb_enc = np.zeros((constants.CB_WIDTH, constants.BIT_WIDTH), dtype=np.float32)
    cb = block_data_dict["coinbase"]
    cb_idx = len(cb)
    fill_idx = 0
    while cb_idx > 0:
        seq_str = cb[cb_idx - constants.HEX_WIDTH : cb_idx]
        if seq_str:
            seq_num = int(seq_str, 16)
            fill_digits(seq_num, cb_enc, fill_idx, constants.BIT_BASE)
        fill_idx += 1
        cb_idx -= constants.HEX_WIDTH
    cb_enc = cb_enc[::-1]
    enc = np.concatenate((nonce_enc, cb_enc), axis=0)
    return np.reshape(enc, (-1,))


def load_bit_data(block_data_path: str, for_train: bool) -> tf.data.Dataset:
    """Load training data from disk."""

    def _data_generator():
        begin_idx = random.randint(0, 100_000)
        value = 342879491
        while True:
            with open(block_data_path, encoding="utf-8") as fp:
                reader = jsonlines.Reader(fp)
                for row_idx, block in enumerate(reader):
                    if row_idx < begin_idx:
                        continue

                    begin_idx = 0
                    if for_train and row_idx % 5 == 0:
                        continue
                    if (not for_train) and row_idx % 5:
                        continue

                    if random.random() < 0.1:
                        value -= 1

                    txs = block["tx"]
                    tx_parts = txs_to_parts(txs)
                    txs_parts_input = txs_to_bits(tx_parts, constants.PARTS_LEN)
                    tx_mask_input = txs_to_mask(tx_parts, constants.PARTS_LEN)
                    block["coinbasevalue"] = value
                    other_input, hb_len = block_to_bits(block)
                    yield txs_parts_input, tx_mask_input, other_input, hb_len * 4
                begin_idx = 0

    dataset = tf.data.Dataset.from_generator(
        _data_generator,
        output_signature=(
            tf.TensorSpec(
                shape=(constants.PARTS_LEN, constants.HASH_WIDTH), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(constants.PARTS_LEN,), dtype=tf.bool),
            tf.TensorSpec(
                shape=(constants.INPUT_HEIGHT, constants.BIT_WIDTH), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    return dataset
