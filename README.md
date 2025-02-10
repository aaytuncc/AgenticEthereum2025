# AI Portfolio Mesh - Learning Service

A decentralized, AI-driven portfolio management system built on **Ethereum mainnet**. This service enables users to **automate portfolio rebalancing** using Safe multisig and Uniswap, with AI agents analyzing market trends and executing trades based on consensus.

Developers can create **autonomous agents** that fetch market data, analyze trends using OpenAI, and interact with smart contracts for rebalancing. This creates a **decentralized agent hub** where developers can monetize high-performing agents while users optimize their portfolios seamlessly.

## System requirements

- Python `>=3.10`
- [Tendermint](https://docs.tendermint.com/v0.34/introduction/install.html) `==0.34.19`
- [IPFS node](https://docs.ipfs.io/install/command-line/#official-distributions) `==0.6.0`
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [Poetry](https://python-poetry.org/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Set Docker permissions so you can run containers as non-root user](https://docs.docker.com/engine/install/linux-postinstall/)

## Run your own agent

### Get the code

1. Clone this repo:
    ```
    git clone git@github.com:valory-xyz/academy-learning-service.git
    ```

2. Create the virtual environment:
    ```
    cd academy-learning-service
    poetry shell
    poetry install
    ```

3. Sync packages:
    ```
    autonomy packages sync --update-packages
    ```

### Prepare the data

1. Generate keys for the agents:
    ```
    autonomy generate-key ethereum -n 4
    ```

2. Create `ethereum_private_key.txt` with one of the private keys from `keys.json` (no newline at end).

3. Deploy [Safe on Ethereum mainnet](https://app.safe.global/welcome) with your agent addresses as signers. Set thresholds:
   - Test Safe: 1/4 signers
   - Production Safe: 3/4 signers

4. Create a [Tenderly](https://tenderly.co/) fork of Ethereum mainnet.

5. Fund your agents and Safe with ETH.

6. Copy and configure environment:
    ```
    cp sample.env .env
    ```

### Environment Variables

Autonolas agent service `.env`:
```
ALL_PARTICIPANTS='["0x35c72A4ebcbEa3E90F3885493FB54FB896B56689"]'
SAFE_CONTRACT_ADDRESS=0x38245ec8f8C326152045b578132349Ebbf7a3Fb6
SAFE_CONTRACT_ADDRESS_SINGLE=0xAde2bd82cDc6662bdE8e4FDE5E727B97B2408047

ETHEREUM_LEDGER_RPC=<your-rpc-url>

COINGECKO_API_KEY=CG-xxx
COINMARKETCAP_API_KEY=xxx
THEGRAPH_API_KEY=xxx
ARLI_API_KEY=xxx
OPENAI_API_KEY=xxx

TRANSFER_TARGET_ADDRESS=0x35c72A4ebcbEa3E90F3885493FB54FB896B56689
PORTFOLIO_MANAGER_CONTRACT_ADDRESS=0x64599B490f3FA6D358AF3119bF4E28744E708703
PORTFOLIO_ADDRESS=0x35c72A4ebcbEa3E90F3885493FB54FB896B56689

API_SELECTION=coinmarketcap

// Feature discarded currently
// TOKENS_TO_REBALANCE='["USDC","ETH"]'
// TARGET_PERCENTAGES='[50.0, 50.0]'
// VARIATION_THRESHOLD=10

ON_CHAIN_SERVICE_ID=1
RESET_PAUSE_DURATION=100
RESET_TENDERMINT_AFTER=10
```

Website `.env`:
```
PORTFOLIO_MANAGER_ADDRESS=0x64599B490f3FA6D358AF3119bF4E28744E708703
USDC_ADDRESS=0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
WETH_ADDRESS=0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
RPC_URL=<your-rpc-url>
COINMARKETCAP_API_KEY=xxx
```

### Run a single agent locally

1. Verify `ALL_PARTICIPANTS` contains only 1 address.
2. Run:
    ```
    bash run_agent.sh
    ```

### Run the service (4 agents) via Docker Compose

1. Verify `ALL_PARTICIPANTS` contains 4 addresses.
2. Check Docker is running:
    ```
    docker
    ```
3. Run service:
    ```
    bash run_service.sh
    ```
4. View logs:
    ```
    docker logs -f learningservice_abci_0
    ```

## Tools Used

- [Autonolas](https://olas.network/) - Autonomous service framework
- [TheGraph](https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=Query&chain=arbitrum-one) - Blockchain data indexing
- [Safe](https://app.safe.global/transactions/queue?safe=eth:0xAde2bd82cDc6662bdE8e4FDE5E727B97B2408047) - Multisig wallet
- [Tenderly](https://virtual.mainnet.rpc.tenderly.co/0f21f795-ccb6-4dd7-98be-5b42b2540a64) - Development environment
- [Uniswap V3](https://docs.uniswap.org/contracts/v3/reference/deployments/) - DEX integration
- [OpenAI](https://openai.com/) - AI analysis


## Conclusion

This Proof of Concept (PoC) represents a groundbreaking step in AI-powered decentralized finance (DeFi). By integrating autonomous agents, AI-driven market analysis, and decentralized smart contract execution, it paves the way for a trustless, data-driven financial ecosystem. The ability to optimize portfolios dynamically using AI insights and execute them securely through Safe multisig ensures both efficiency and transparency.

The potential for this system extends far beyond its current implementation. Future iterations could involve enhanced AI models, multi-chain support, and agent-driven financial strategies, further redefining how DeFi portfolio management operates. This PoC demonstrates the viability of a decentralized, AI-driven financial system—a promising innovation for the future of Web3 and blockchain automation. 🚀