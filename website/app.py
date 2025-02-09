from flask import Flask, render_template, jsonify, request
from blockchain import BlockchainManager
import os
from dotenv import load_dotenv
from decimal import Decimal
from web3 import Web3
import requests

load_dotenv()  # Load environment variables

app = Flask(__name__)

# Token decimals
USDC_DECIMALS = 6
WETH_DECIMALS = 18

# Initialize blockchain connection
blockchain = BlockchainManager(os.getenv('RPC_URL'))

# Load Portfolio Manager contract
blockchain.load_contract(
    'portfolio_manager',
    os.getenv('PORTFOLIO_MANAGER_ADDRESS'),
    'PortfolioManager.json'
)

# Load IERC20 contract
blockchain.load_contract(
    'ierc20',
    '',  # Address will be set dynamically based on token
    'IERC20.json'
)

@app.route('/')
def index():
    return render_template('index.html', config={
        'USDC_ADDRESS': os.getenv('USDC_ADDRESS'),
        'WETH_ADDRESS': os.getenv('WETH_ADDRESS')
    })

# Example endpoint to read from blockchain
@app.route('/api/blockchain/read/<method_name>')
def read_blockchain(method_name):
    try:
        result = blockchain.read_contract(method_name)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Example endpoint to write to blockchain
@app.route('/api/blockchain/write/<method_name>', methods=['POST'])
def write_blockchain(method_name):
    try:
        data = request.json
        result = blockchain.write_contract(
            method_name,
            sender_address=os.getenv('SENDER_ADDRESS'),
            private_key=os.getenv('PRIVATE_KEY'),
            *data.get('args', [])
        )
        return jsonify({'success': True, 'transaction': result.hex()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/approve_and_deposit', methods=['POST'])
def approve_and_deposit():
    try:
        data = request.json
        token_address = data['tokenAddress']
        amount = data['amount']
        user_address = data['userAddress']
        
        # Determine decimals based on token
        decimals = USDC_DECIMALS if token_address.lower() == os.getenv('USDC_ADDRESS').lower() else WETH_DECIMALS
        
        # Calculate amount with decimals
        scaled_amount = int(Decimal(amount) * Decimal(10 ** decimals))
        
        # Get contract instances
        portfolio_manager = blockchain.get_contract('portfolio_manager')
        
        # Update token contract address
        blockchain.contracts['ierc20'].address = token_address
        token_contract = blockchain.get_contract('ierc20')
        
        return jsonify({
            'success': True,
            'approveData': {
                'to': token_address,
                'data': token_contract.encodeABI('approve', [portfolio_manager.address, scaled_amount])
            },
            'depositData': {
                'to': portfolio_manager.address,
                'data': portfolio_manager.encodeABI('deposit', [token_address, scaled_amount])
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/withdraw', methods=['POST'])
def withdraw():
    try:
        data = request.json
        token_address = data['tokenAddress']
        amount = data['amount']
        user_address = data['userAddress']
        
        # Determine decimals based on token
        decimals = USDC_DECIMALS if token_address.lower() == os.getenv('USDC_ADDRESS').lower() else WETH_DECIMALS
        
        # Calculate amount with decimals
        scaled_amount = int(Decimal(amount) * Decimal(10 ** decimals))
        
        # Get contract instance
        portfolio_manager = blockchain.get_contract('portfolio_manager')
        
        return jsonify({
            'success': True,
            'withdrawData': {
                'to': portfolio_manager.address,
                'data': portfolio_manager.encodeABI('withdraw', [token_address, scaled_amount])
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/balances', methods=['GET'])
def get_balances():
    try:
        user_address = request.args.get('address')
        if not user_address:
            return jsonify({'success': False, 'error': 'No address provided'}), 400

        portfolio_manager = blockchain.get_contract('portfolio_manager')
        
        # Check if user is registered
        is_registered = portfolio_manager.functions.portfolios(
            Web3.to_checksum_address(user_address)
        ).call()
        
        if not is_registered:
            return jsonify({'success': False, 'error': 'User not registered'}), 400
            
        token_addresses = [
            Web3.to_checksum_address(os.getenv('USDC_ADDRESS')),
            Web3.to_checksum_address(os.getenv('WETH_ADDRESS'))
        ]
        
        balances = portfolio_manager.functions.getUserBalances(
            Web3.to_checksum_address(user_address),
            token_addresses
        ).call()
        
        usdc_formatted = balances[0] / (10 ** USDC_DECIMALS)
        weth_formatted = balances[1] / (10 ** WETH_DECIMALS)
        
        # Fetch token prices from CoinMarketCap
        headers = {
            'X-CMC_PRO_API_KEY': os.getenv('COINMARKETCAP_API_KEY'),
            'Accept': 'application/json'
        }
        
        # Using ETH price for WETH
        response = requests.get(
            'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest',
            params={'symbol': 'ETH,USDC'},
            headers=headers
        )
        
        price_data = response.json()
        weth_price = price_data['data']['ETH'][0]['quote']['USD']['price']
        usdc_price = price_data['data']['USDC'][0]['quote']['USD']['price']
        
        # Calculate USD values
        usdc_value = usdc_formatted * usdc_price
        weth_value = weth_formatted * weth_price
        total_value = usdc_value + weth_value
        
        return jsonify({
            'success': True,
            'balances': {
                'usdc': usdc_formatted,
                'weth': weth_formatted
            },
            'usd_values': {
                'usdc': usdc_value,
                'weth': weth_value,
                'total': total_value
            }
        })
    except Exception as e:
        print(f"Error in get_balances: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)