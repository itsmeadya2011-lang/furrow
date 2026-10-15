import pytest
from blockchain import Blockchain


def test_blockchain_has_genesis_block():
    """A newly initialized blockchain should contain a genesis block."""
    blockchain = Blockchain()
    assert len(blockchain.chain) == 1
    genesis_block = blockchain.chain[0]
    assert genesis_block.index == 0
    assert genesis_block.previous_hash == "0"