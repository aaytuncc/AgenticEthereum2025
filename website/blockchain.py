from web3 import Web3
from typing import Dict
import json
import os

class BlockchainManager:
    def __init__(self, rpc_url: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contracts: Dict[str, any] = {}
        
    def connect(self) -> bool:
        return self.w3.is_connected()
    
    def load_contract(self, name: str, contract_address: str, abi_filename: str):
        try:
            # Get absolute path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            abi_path = os.path.join(
                current_dir,
                'contracts',
                'abis',
                abi_filename
            )
            
            # Debug prints
            print(f"Looking for ABI file at: {abi_path}")
            print(f"File exists: {os.path.exists(abi_path)}")
            
            with open(abi_path, 'r') as file:
                contract_abi = json.load(file)
            
            self.contracts[name] = self.w3.eth.contract(
                address=contract_address,
                abi=contract_abi
            )
        except Exception as e:
            print(f"Error loading contract: {str(e)}")
            raise
        
    def get_contract(self, name: str):
        return self.contracts.get(name)
    
    def read_contract(self, method_name: str, *args):
        if not self.contracts:
            raise Exception("Contracts not loaded")
            
        # Get contract method
        method = getattr(self.contracts.values(), method_name)
        return method(*args).call()
    
    def write_contract(self, method_name: str, sender_address: str, private_key: str, *args):
        if not self.contracts:
            raise Exception("Contracts not loaded")
            
        # Get contract method
        method = getattr(self.contracts.values(), method_name)
        
        # Build transaction
        transaction = method(*args).build_transaction({
            'from': sender_address,
            'nonce': self.w3.eth.get_transaction_count(sender_address),
            'gas': 2000000,  # Adjust gas limit as needed
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)