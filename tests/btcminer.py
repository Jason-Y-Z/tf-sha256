import hashlib
import struct
from typing import Any, Dict


def int2lehex(value, width):
    """
    Convert an unsigned integer to a little endian ASCII hex string.

    Args:
        value (int): value
        width (int): byte width

    Returns:
        string: ASCII hex string
    """

    return value.to_bytes(width, byteorder="little").hex()


def int2varinthex(value):
    """
    Convert an unsigned integer to little endian varint ASCII hex string.

    Args:
        value (int): value

    Returns:
        string: ASCII hex string
    """

    if value < 0xFD:
        return int2lehex(value, 1)
    elif value <= 0xFFFF:
        return "fd" + int2lehex(value, 2)
    elif value <= 0xFFFFFFFF:
        return "fe" + int2lehex(value, 4)
    else:
        return "ff" + int2lehex(value, 8)


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

    height_enc = bytes([width]).hex() + int2lehex(height, width)
    if return_len:
        return height_enc, len(height_enc)
    else:
        return height_enc


def tx_compute_merkle_root(tx_hashes):
    """
    Compute the Merkle Root of a list of transaction hashes.

    Arguments:
        tx_hashes (list): list of transaction hashes as ASCII hex strings

    Returns:
        string: merkle root as a big endian ASCII hex string
    """

    # Convert list of ASCII hex transaction hashes into bytes
    tx_hashes = [bytes.fromhex(tx_hash)[::-1] for tx_hash in tx_hashes]

    # Iteratively compute the merkle root hash
    while len(tx_hashes) > 1:

        # Duplicate last hash if the list is odd
        if len(tx_hashes) % 2 != 0:
            tx_hashes.append(tx_hashes[-1])

        tx_hashes_new = []

        for _ in range(len(tx_hashes) // 2):
            # Concatenate the next two
            concat = tx_hashes.pop(0) + tx_hashes.pop(0)
            # Hash them
            concat_hash = hashlib.sha256(hashlib.sha256(concat).digest()).digest()
            # Add them to our working list
            tx_hashes_new.append(concat_hash)
        tx_hashes = tx_hashes_new

    # Format the root in big endian ascii hex
    return tx_hashes[0][::-1].hex()


def tx_make_coinbase(
    coinbase_script: str, address: str, value: int, height: int
) -> str:
    """
    Create a coinbase transaction.

    Arguments:
        coinbase_script (string): arbitrary script as an ASCII hex string
        address (string): Base58 Bitcoin address
        value (int): coinbase value
        height (int): mined block height

    Returns:
        string: coinbase transaction as an ASCII hex string
    """

    # See https://en.bitcoin.it/wiki/Transaction
    coinbase_script = tx_encode_coinbase_height(height) + coinbase_script

    # Create a pubkey script
    # OP_DUP OP_HASH160 <len to push> <pubkey> OP_EQUALVERIFY OP_CHECKSIG
    pubkey_script = "76" + "a9" + "14" + bitcoinaddress2hash160(address) + "88" + "ac"

    tx_front = ""
    # version
    tx_front += "01000000"
    # in-counter
    tx_front += "01"
    # input[0] prev hash
    tx_front += "0" * 64
    # input[0] prev seqnum
    tx_front += "ffffffff"
    # input[0] script len
    tx_front += int2varinthex(len(coinbase_script) // 2)

    # input[0] script
    tx_mid = coinbase_script
    # input[0] seqnum
    tx_back = "ffffffff"
    # out-counter
    tx_back += "01"
    # output[0] value
    tx_back += int2lehex(value, 8)
    # output[0] script len
    tx_back += int2varinthex(len(pubkey_script) // 2)
    # output[0] script
    tx_back += pubkey_script
    # lock-time
    tx_back += "00000000"

    return tx_front + tx_mid + tx_back


def tx_compute_hash(tx: str) -> str:
    """
    Compute the SHA256 double hash of a transaction.

    Arguments:
        tx (string): transaction data as an ASCII hex string

    Return:
        string: transaction hash as an ASCII hex string
    """

    return (
        hashlib.sha256(hashlib.sha256(bytes.fromhex(tx)).digest()).digest()[::-1].hex()
    )


def construct_coinbase(
    block_template: Dict[str, Any],
    address: str,
    extranonce: int | None = None,
    coinbase_message: str = "coinbase message".encode().hex(),
    coinbase_script: str | None = None,
):
    """Update the coinbase transaction with the new extra nonce."""
    coinbase_tx = block_template["transactions"][0]
    if coinbase_script is None:
        coinbase_script: str = coinbase_message + int2lehex(extranonce, 4)
    coinbase_tx["data"] = tx_make_coinbase(
        coinbase_script,
        address,
        block_template["coinbasevalue"],
        block_template["height"],
    )
    coinbase_tx["hash"] = tx_compute_hash(coinbase_tx["data"])


def block_bits2target(bits):
    """
    Convert compressed target (block bits) encoding to target value.

    Arguments:
        bits (string): compressed target as an ASCII hex string

    Returns:
        bytes: big endian target
    """

    # Bits: 1b0404cb
    #       1b          left shift of (0x1b - 3) bytes
    #         0404cb    value
    bits = bytes.fromhex(bits)
    shift = bits[0] - 3
    value = bits[1:]

    # Shift value to the left by shift
    target = value + b"\x00" * shift
    # Add leading zeros
    target = b"\x00" * (32 - len(target)) + target

    return target


def block_make_header(block):
    """
    Make the block header.

    Arguments:
        block (dict): block template

    Returns:
        bytes: block header
    """

    # Version
    version = struct.pack("<L", int(block["version"]))
    # Previous Block Hash
    pbh = bytes.fromhex(block["previousblockhash"])[::-1]
    # Merkle Root Hash
    merkleroot = bytes.fromhex(block["merkleroot"])[::-1]
    # Time
    curtime = struct.pack("<L", block["curtime"])
    # Target Bits
    bits = bytes.fromhex(block["bits"])[::-1]
    # Nonce
    nonce = struct.pack("<L", block["nonce"])

    return version + pbh + merkleroot + curtime + bits + nonce


def fill_coinbase_and_nonce(
    blk_temp: dict, address: str, coinbase: str, nonce: int
) -> tuple[bytes, bytes]:
    """Fill the given coinbase and nonce into the block template."""
    blk_temp["nonce"] = nonce
    blk_temp["transactions"].insert(0, {})
    construct_coinbase(blk_temp, address, coinbase_script=coinbase)
    blk_temp["merkleroot"] = tx_compute_merkle_root(
        [tx["hash"] for tx in blk_temp["transactions"]]
    )
    block_header = block_make_header(blk_temp)
    target_hash = block_bits2target(blk_temp["bits"])
    return block_header, target_hash
