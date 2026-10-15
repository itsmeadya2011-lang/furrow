import hashlib
import json
import time
from typing import Any


class Block:
    def __init__(self, index: int, timestamp: float, data: list[dict[str, Any]], 
                 previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate the hash of the block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int):
        """Mine the block using proof-of-work."""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"Block mined: {self.hash}")


class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain: list[Block] = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.pending_transactions: list[dict[str, Any]] = []
        self.mining_reward = 100

    def create_genesis_block(self) -> Block:
        """Create the genesis block."""
        return Block(0, time.time(), [{"genesis": "block"}], "0")

    def get_latest_block(self) -> Block:
        """Get the latest block in the chain."""
        return self.chain[-1]

    def mine_pending_transactions(self, mining_reward_address: str):
        """Mine all pending transactions and reward the miner."""
        # Add mining reward transaction
        reward_transaction = {
            "from": "network",
            "to": mining_reward_address,
            "amount": self.mining_reward
        }
        self.pending_transactions.append(reward_transaction)

        # Create a new block with pending transactions
        block = Block(
            len(self.chain),
            time.time(),
            self.pending_transactions,
            self.get_latest_block().hash
        )

        # Mine the block
        block.mine_block(self.difficulty)

        # Add the block to the chain
        self.chain.append(block)
        
        # Reset pending transactions
        self.pending_transactions = []
        print(f"Block successfully mined! Reward sent to {mining_reward_address}")

    def create_transaction(self, sender: str, recipient: str, amount: float):
        """Create a new transaction."""
        transaction = {
            "from": sender,
            "to": recipient,
            "amount": amount
        }
        self.pending_transactions.append(transaction)
        print(f"Transaction added: {sender} -> {recipient}: {amount}")

    def get_balance(self, address: str) -> float:
        """Get the balance of an address."""
        balance = 0
        for block in self.chain:
            for transaction in block.data:
                # Skip genesis block transaction which has different structure
                if "from" not in transaction or "to" not in transaction:
                    continue
                if transaction["from"] == address:
                    balance -= transaction["amount"]
                if transaction["to"] == address:
                    balance += transaction["amount"]
        return balance

    def is_chain_valid(self) -> bool:
        """Check if the blockchain is valid."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # Check if the current block's hash is correct
            if current_block.hash != current_block.calculate_hash():
                print(f"Invalid hash at block {i}")
                return False

            # Check if the current block points to the previous block
            if current_block.previous_hash != previous_block.hash:
                print(f"Invalid previous hash at block {i}")
                return False

            # Check if the block satisfies proof-of-work
            target = "0" * self.difficulty
            if current_block.hash[:self.difficulty] != target:
                print(f"Invalid proof-of-work at block {i}")
                return False

        return True

    def print_chain(self):
        """Print the entire blockchain."""
        for block in self.chain:
            print(f"Block {block.index}:")
            print(f"  Timestamp: {block.timestamp}")
            print(f"  Data: {json.dumps(block.data, indent=2)}")
            print(f"  Previous Hash: {block.previous_hash}")
            print(f"  Hash: {block.hash}")
            print(f"  Nonce: {block.nonce}")
            print()


if __name__ == "__main__":
    # Create a blockchain
    my_blockchain = Blockchain(difficulty=4)

    print("Starting the miner...")
    my_blockchain.mine_pending_transactions("miner1_address")

    print("\nCreating some transactions...")
    my_blockchain.create_transaction("miner1_address", "alice", 50)
    my_blockchain.create_transaction("bob", "alice", 25)
    my_blockchain.create_transaction("alice", "bob", 10)

    print("\nStarting the miner again...")
    my_blockchain.mine_pending_transactions("miner2_address")

    print("\nCreating more transactions...")
    my_blockchain.create_transaction("bob", "charlie", 15)
    my_blockchain.create_transaction("alice", "david", 30)

    print("\nStarting the miner again...")
    my_blockchain.mine_pending_transactions("miner1_address")

    print("\n=== Balances ===")
    print(f"Miner1: {my_blockchain.get_balance('miner1_address')}")
    print(f"Miner2: {my_blockchain.get_balance('miner2_address')}")
    print(f"Alice: {my_blockchain.get_balance('alice')}")
    print(f"Bob: {my_blockchain.get_balance('bob')}")
    print(f"Charlie: {my_blockchain.get_balance('charlie')}")
    print(f"David: {my_blockchain.get_balance('david')}")

    print("\n=== Blockchain Validity ===")
    print(f"Is blockchain valid? {my_blockchain.is_chain_valid()}")

    print("\n=== Blockchain Contents ===")
    my_blockchain.print_chain()