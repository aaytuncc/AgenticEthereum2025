# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This package contains round behaviours of LearningAbciApp."""

import json
import requests
from abc import ABC
from pathlib import Path
from tempfile import mkdtemp
from typing import Union, Tuple, Dict, Generator, Optional, Set, Type, cast
from datetime import datetime

from packages.valory.contracts.erc20.contract import ERC20
from packages.valory.contracts.mock_dex.contract import MOCKDEX
from packages.valory.contracts.portfolio_manager.contract import PORTFOLIOMANAGER



from packages.valory.contracts.gnosis_safe.contract import (
    GnosisSafeContract,
    SafeOperation,
)
from packages.valory.contracts.multisend.contract import (
    MultiSendContract,
    MultiSendOperation,
)
from packages.valory.protocols.contract_api import ContractApiMessage
from packages.valory.protocols.ledger_api import LedgerApiMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.abstract_round_abci.io_.store import SupportedFiletype
from packages.valory.skills.learning_abci.models import (
    CoingeckoSpecs,
    ArliSpecs,
    CoinMarketCapSpecs,
    TheGraphSpecs,
    Params,
    SharedState,
    OpenAISpecs,
)
from packages.valory.skills.learning_abci.payloads import (
    ApiSelectionPayload,
    AlternativeDataPullPayload,
    DataPullPayload,
    DecisionMakingPayload,
    TxPreparationPayload,
)
from packages.valory.skills.learning_abci.rounds import (
    ApiSelectionRound,
    AlternativeDataPullRound,
    DataPullRound,
    DecisionMakingRound,
    Event,
    LearningAbciApp,
    SynchronizedData,
    TxPreparationRound,
)
from packages.valory.skills.transaction_settlement_abci.payload_tools import (
    hash_payload_to_hex,
)
from packages.valory.skills.transaction_settlement_abci.rounds import TX_HASH_LENGTH


# Define some constants
ZERO_VALUE = 0
HTTP_OK = 200
GNOSIS_CHAIN_ID = "gnosis"
ETHEREUM_CHAIN_ID = "ethereum"
EMPTY_CALL_DATA = b"0x"
SAFE_GAS = 0
VALUE_KEY = "value"
TO_ADDRESS_KEY = "to_address"
METADATA_FILENAME = "metadata.json"
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2" 

class LearningBaseBehaviour(BaseBehaviour, ABC):  # pylint: disable=too-many-ancestors
    """Base behaviour for the learning_abci behaviours."""

    @property
    def params(self) -> Params:
        """Return the params. Configs go here"""
        return cast(Params, super().params)

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data. This data is common to all agents"""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def local_state(self) -> SharedState:
        """Return the local state of this particular agent."""
        return cast(SharedState, self.context.state)

    @property
    def coingecko_specs(self) -> CoingeckoSpecs:
        """Get the Coingecko api specs."""
        return self.context.coingecko_specs

    @property
    def arli_specs(self) -> ArliSpecs:
        """Get the Coingecko api specs."""
        
        return self.context.arli_specs

    @property
    def openai_specs(self) -> OpenAISpecs:
        """Get the OpenAI api specs."""
        return self.context.openai_specs

    @property
    def coinmarketcap_specs(self) -> CoinMarketCapSpecs:
        """Get the CoinMarketCap api specs."""
        return self.context.coinmarketcap_specs
    
    @property
    def thegraph_specs(self) -> TheGraphSpecs:
        """Get the TheGraph api specs."""
        return self.context.thegraph_specs

    @property
    def metadata_filepath(self) -> str:
        """Get the temporary filepath to the metadata."""
        return str(Path(mkdtemp()) / METADATA_FILENAME)

    def get_sync_timestamp(self) -> float:
        """Get the synchronized time from Tendermint's last block."""
        now = cast(
            SharedState, self.context.state
        ).round_sequence.last_round_transition_timestamp.timestamp()

        return now

class ApiSelectionBehaviour(
    LearningBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """ApiSelectionBehaviour
    A behavior for selecting an API source, determining whether to use "coingecko" or "coinmarketcap" based on parameters, and submitting the decision.

    Sets api_selection to "coingecko" by default, or "coinmarketcap" if specified in parameters.
    """

    matching_round: Type[AbstractRound] = ApiSelectionRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address

            selection = self.params.api_selection_string
            api_selection = "coingecko"
            if selection == "coinmarketcap":
                api_selection = selection

            payload = ApiSelectionPayload(sender=sender, api_selection=api_selection)
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

class DataPullBehaviour(LearningBaseBehaviour):  # pylint: disable=too-many-ancestors
    """This behaviours pulls token prices from API endpoints """

    matching_round: Type[AbstractRound] = DataPullRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address

            token_values, total_portfolio_value = yield from self.calculate_portfolio_allocation()
            self.context.logger.info(f"Token values: {token_values}")
            self.context.logger.info(f"Total portfolio value: {total_portfolio_value}")

            # Convert token values to JSON, dict is not hashable causing problems
            token_values_json = json.dumps(token_values, sort_keys=True) if token_values else None
            self.context.logger.info(f"Token values JSON: {token_values_json}")


            # Prepare the payload to be shared with other agents
            payload = DataPullPayload(
                sender=sender,
                token_values=token_values_json,
                total_portfolio_value=total_portfolio_value,
            )

        # Send the payload to all agents and mark the behaviour as done
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_token_price_specs(self, symbol) -> Generator[None, None, Optional[float]]:
        """Get token price from Coingecko using ApiSpecs"""

        # Get a copy of the specs and update based on the symbol
        specs = self.coingecko_specs.get_spec()  # Get a dictionary instead of assuming a specs attribute
        if symbol == "ETH":
            specs["parameters"]["ids"] = "ethereum"
            response_key = "ethereum"
        elif symbol == "USDC":
            specs["parameters"]["ids"] = "usd-coin"
            response_key = "usd-coin"
        else:
            self.context.logger.error(f"Unsupported token symbol: {symbol}")
            return None

        # Make the HTTP request without modifying self.coingecko_specs directly
        raw_response = yield from self.get_http_response(**specs)

        # Process the response using response_key
        response = self.coingecko_specs.process_response(raw_response)
        price = response.get(response_key, {}).get("usd", None)
        
        self.context.logger.info(f"Got token price from Coingecko: {price}")
        return price
 
    def get_token_balances(self) -> Generator[None, None, Optional[Dict[str, float]]]:
        """
        Get balances for each specified token from the deployed contract using parameters from self.params.

        :return: Dictionary of token balances, or None if an error occurs.
        """
        self.context.logger.info("Starting to fetch token balances for the portfolio.")

        # Retrieve portfolio address and tokens to rebalance from params
        portfolio_address = self.params.portfolio_address_string
        portfolio_manager_contract_address = self.params.portfolio_manager_contract_address_string

        # Use default token addresses and their decimals
        tokens_to_rebalance = {
            "USDC": {"address": USDC_ADDRESS, "decimals": 6},
            "WETH": {"address": WETH_ADDRESS, "decimals": 18}
        }

        # Log the portfolio details and tokens
        self.context.logger.info(f"Portfolio Address: {portfolio_address}")
        self.context.logger.info(f"Portfolio Manager Contract: {portfolio_manager_contract_address}")
        self.context.logger.info(f"Tokens to Rebalance: {tokens_to_rebalance}")

        # Prepare list of token addresses in the same order as the symbols
        token_addresses = [token_info["address"] for token_info in tokens_to_rebalance.values()]
        token_symbols = list(tokens_to_rebalance.keys())

        # Call the contract API to get all token balances at once
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=portfolio_manager_contract_address,
            contract_id=str(PORTFOLIOMANAGER.contract_id),
            contract_callable="getUserBalances",
            user=portfolio_address,
            tokens=token_addresses,
            chain_id=ETHEREUM_CHAIN_ID,
        )

        # Check if the response contains the expected balance data
        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(f"Error retrieving balances: {response_msg}")
            return None

        # Extract balances from the response
        balances_list = response_msg.raw_transaction.body.get("balances", None)
        if balances_list is None:
            self.context.logger.error("No balance data returned")
            return None

        # Create dictionary mapping token symbols to their balances
        balances = {}
        for i, (symbol, balance) in enumerate(zip(token_symbols, balances_list)):
            if balance is not None:
                # Convert the balance using the correct number of decimals for each token
                decimals = tokens_to_rebalance[symbol]["decimals"]
                readable_balance = float(balance) / (10 ** decimals)
                balances[symbol] = readable_balance
                self.context.logger.info(f"Balance for {symbol} (in readable format): {readable_balance}")
            else:
                self.context.logger.error(f"No balance data returned for {symbol}")
                balances[symbol] = None

        # Final printout of all token balances
        self.context.logger.info("Completed fetching balances for all tokens.")
        for token, balance in balances.items():
            if balance is not None:
                self.context.logger.info(f"Final balance for {token}: {balance}")
            else:
                self.context.logger.info(f"Balance for {token} could not be retrieved.")

        return balances if balances else None

    def calculate_portfolio_allocation(self) -> Generator[None, None, Optional[Tuple[Dict[str, float], float]]]:
        """
        Calculate the total portfolio value and percentage allocation based on token balances and prices.

        :return: A tuple containing:
                - token_values: Dictionary of each token's value in USD.
                - total_portfolio_value: Total value of the portfolio in USD.
        """

        # Step 1: Get token balances
        self.context.logger.info("Fetching token balances...")
        token_balances = yield from self.get_token_balances()
        if token_balances is None:
            self.context.logger.error("Failed to retrieve token balances.")
            return None

        # Step 2: Initialize total value
        total_portfolio_value = 0.0
        token_values = {}

        # Step 3: Get prices and calculate value for each token
        for token_symbol, balance in token_balances.items():
            if balance is None:
                self.context.logger.error(f"No balance available for {token_symbol}")
                continue

            # Fetch token price
            self.context.logger.info(f"Fetching price for {token_symbol}...")
            price = yield from self.get_token_price_specs(symbol=token_symbol)
            if price is None:
                self.context.logger.error(f"Failed to retrieve price for {token_symbol}")
                continue

            # Calculate token's value in the portfolio
            token_value = balance * price
            token_values[token_symbol] = token_value
            total_portfolio_value += token_value

            self.context.logger.info(f"Value for {token_symbol}: {token_value:.2f} USD")

        # Step 4: Calculate percentage allocation
        if total_portfolio_value == 0:
            self.context.logger.error("Total portfolio value is zero; cannot calculate allocation.")
            return None

        self.context.logger.info("Portfolio Allocation:")
        for token_symbol, token_value in token_values.items():
            percentage = (token_value / total_portfolio_value) * 100
            self.context.logger.info(f"{token_symbol}: {percentage:.2f}% of portfolio (Value: {token_value:.2f} USD)")

        self.context.logger.info(f"Total Portfolio Value: {total_portfolio_value:.2f} USD")

        # # Step 5: Generate and store the rebalancing report in IPFS
        # report_ipfs_hash = yield from self.generate_and_store_report(token_values, total_portfolio_value)
        # if report_ipfs_hash:
        #     self.context.logger.info(f"Rebalancing report stored in IPFS: https://gateway.autonolas.tech/ipfs/{report_ipfs_hash}")
        # else:
        #     self.context.logger.error("Failed to store rebalancing report in IPFS.")


        # Return token values and total portfolio value
        return token_values, total_portfolio_value
        

class AlternativeDataPullBehaviour(LearningBaseBehaviour):  # pylint: disable=too-many-ancestors
    """This behaviours pulls token prices from API endpoints and reads the native balance of an account"""

    matching_round: Type[AbstractRound] = AlternativeDataPullRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address

            token_values, total_portfolio_value = yield from self.calculate_portfolio_allocation()
            self.context.logger.info(f"Token values: {token_values}")
            self.context.logger.info(f"Total portfolio value: {total_portfolio_value}")


            # Convert token values to JSON, dict is not hashable causing problems
            token_values_json = json.dumps(token_values, sort_keys=True) if token_values else None
            self.context.logger.info(f"Token values JSON: {token_values_json}")

            # Prepare the payload to be shared with other agents
            payload = AlternativeDataPullPayload(
                sender=sender,
                token_values=token_values_json,
                total_portfolio_value=total_portfolio_value,
            )

        # Send the payload to all agents and mark the behaviour as done
        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_token_price_specs(self, symbol) -> Generator[None, None, Optional[float]]:
        """Get token price from Coinmarketcap using ApiSpecs"""

        # Get the specs
        specs = self.coinmarketcap_specs.get_spec()
        specs["parameters"]["symbol"] = symbol

        # Make the call
        raw_response = yield from self.get_http_response(**specs)

        # Process the response
        response = self.coinmarketcap_specs.process_response(raw_response)

        # Navigate to get the price
        token_data = response.get(symbol, {})
        price_info = token_data.get("quote", {}).get("USD", {})
        price = price_info.get("price", None)

        # Log and return the price
        self.context.logger.info(f"Got token price from CoinMarketCap: {price}")

        return price

    def get_token_balances(self) -> Generator[None, None, Optional[Dict[str, float]]]:
        """
        Get balances for each specified token from the deployed contract using parameters from self.params.

        :return: Dictionary of token balances, or None if an error occurs.
        """
        self.context.logger.info("Starting to fetch token balances for the portfolio.")

        # Retrieve portfolio address and tokens to rebalance from params
        portfolio_address = self.params.portfolio_address_string
        portfolio_manager_contract_address = self.params.portfolio_manager_contract_address_string

        # Use default token addresses and their decimals
        tokens_to_rebalance = {
            "USDC": {"address": USDC_ADDRESS, "decimals": 6},
            "WETH": {"address": WETH_ADDRESS, "decimals": 18}
        }

        # Log the portfolio details and tokens
        self.context.logger.info(f"Portfolio Address: {portfolio_address}")
        self.context.logger.info(f"Portfolio Manager Contract: {portfolio_manager_contract_address}")
        self.context.logger.info(f"Tokens to Rebalance: {tokens_to_rebalance}")

        # Prepare list of token addresses in the same order as the symbols
        token_addresses = [token_info["address"] for token_info in tokens_to_rebalance.values()]
        token_symbols = list(tokens_to_rebalance.keys())

        # Call the contract API to get all token balances at once
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=portfolio_manager_contract_address,
            contract_id=str(PORTFOLIOMANAGER.contract_id),
            contract_callable="get_user_balances",
            user=portfolio_address,
            tokens=token_addresses,
            chain_id=ETHEREUM_CHAIN_ID,
        )

        # Check if the response contains the expected balance data
        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(f"Error retrieving balances: {response_msg}")
            return None

        # Extract balances from the response
        balances_list = response_msg.raw_transaction.body.get("balances", None)
        if balances_list is None:
            self.context.logger.error("No balance data returned")
            return None

        # Create dictionary mapping token symbols to their balances
        balances = {}
        for i, (symbol, balance) in enumerate(zip(token_symbols, balances_list)):
            if balance is not None:
                # Convert the balance using the correct number of decimals for each token
                decimals = tokens_to_rebalance[symbol]["decimals"]
                readable_balance = float(balance) / (10 ** decimals)
                balances[symbol] = readable_balance
                self.context.logger.info(f"Balance for {symbol} (in readable format): {readable_balance}")
            else:
                self.context.logger.error(f"No balance data returned for {symbol}")
                balances[symbol] = None

        # Final printout of all token balances
        self.context.logger.info("Completed fetching balances for all tokens.")
        for token, balance in balances.items():
            if balance is not None:
                self.context.logger.info(f"Final balance for {token}: {balance}")
            else:
                self.context.logger.info(f"Balance for {token} could not be retrieved.")

        return balances if balances else None

    def calculate_portfolio_allocation(self) -> Generator[None, None, Optional[Tuple[Dict[str, float], float]]]:
        """
        Calculate the total portfolio value and percentage allocation based on token balances and prices.

        :return: A tuple containing:
                - token_values: Dictionary of each token's value in USD.
                - total_portfolio_value: Total value of the portfolio in USD.
        """

        # Step 1: Get token balances
        self.context.logger.info("Fetching token balances...")
        token_balances = yield from self.get_token_balances()
        if token_balances is None:
            self.context.logger.error("Failed to retrieve token balances.")
            return None

        # Step 2: Initialize total value
        total_portfolio_value = 0.0
        token_values = {}

        # Step 3: Get prices and calculate value for each token
        for token_symbol, balance in token_balances.items():
            if balance is None:
                self.context.logger.error(f"No balance available for {token_symbol}")
                continue

            # Fetch token price
            self.context.logger.info(f"Fetching price for {token_symbol}...")
            price = yield from self.get_token_price_specs(symbol=token_symbol)
            if price is None:
                self.context.logger.error(f"Failed to retrieve price for {token_symbol}")
                continue

            # Calculate token's value in the portfolio
            token_value = balance * price
            token_values[token_symbol] = token_value
            total_portfolio_value += token_value

            self.context.logger.info(f"Value for {token_symbol}: {token_value:.2f} USD")

        # Step 4: Calculate percentage allocation
        if total_portfolio_value == 0:
            self.context.logger.error("Total portfolio value is zero; cannot calculate allocation.")
            return None

        self.context.logger.info("Portfolio Allocation:")
        for token_symbol, token_value in token_values.items():
            percentage = (token_value / total_portfolio_value) * 100
            self.context.logger.info(f"{token_symbol}: {percentage:.2f}% of portfolio (Value: {token_value:.2f} USD)")

        self.context.logger.info(f"Total Portfolio Value: {total_portfolio_value:.2f} USD")

        # # Step 5: Generate and store the rebalancing report in IPFS
        # report_ipfs_hash = yield from self.generate_and_store_report(token_values, total_portfolio_value)
        # if report_ipfs_hash:
        #     self.context.logger.info(f"Rebalancing report stored in IPFS: https://gateway.autonolas.tech/ipfs/{report_ipfs_hash}")
        # else:
        #     self.context.logger.error("Failed to store rebalancing report in IPFS.")


        # Return token values and total portfolio value
        return token_values, total_portfolio_value

class DecisionMakingBehaviour(
    LearningBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """DecisionMakingBehaviour"""

    matching_round: Type[AbstractRound] = DecisionMakingRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address

            # Make a decision: either transact or not
            event,adjustment_balances = yield from self.get_next_event()

            payload = DecisionMakingPayload(sender=sender, event=event, adjustment_balances=adjustment_balances)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_uniswap_token_price_specs(self) -> Generator[None, None, Optional[str]]:
        """Get token price from Uniswap V3 using The Graph API and ask LLM if portfolio should be rebalanced."""

        # **Get the specs**
        specs = self.thegraph_specs.get_spec()
        # **GraphQL Query to Fetch Token Price Data**
        graphql_query = """
        {
          USDC: tokenDayDatas(
            where: { token: "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" }
            orderBy: date
            orderDirection: desc
            first: 7
          ) {
            date
            priceUSD
            volumeUSD
            feesUSD
          }
          WETH: tokenDayDatas(
            where: { token: "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2" }
            orderBy: date
            orderDirection: desc
            first: 7
          ) {
            date
            priceUSD
            volumeUSD
            feesUSD
          }
        }
        """

        # **Prepare API request parameters**
        specs["parameters"]["query"] = graphql_query
        specs["parameters"]["operationName"] = "Subgraphs"
        specs["parameters"]["variables"] = {}

        # **Prepare JSON payload**
        content = json.dumps(specs["parameters"]).encode("utf-8")

        # # **Log full request details**
        # self.context.logger.info(f"API Specs being used: {specs}")
        # self.context.logger.info(f"Payload being sent: {json.dumps(specs['parameters'], indent=2)}")
        # self.context.logger.info(f"Payload being sent as content: {content.decode('utf-8')}")

        try:
            # **✅ Fetch token data from Uniswap V3**
            raw_response = yield from self.get_http_response(
                method="POST",
                url=specs["url"],
                content=content,
                headers=specs["headers"]
            )

            # **Check if raw_response is an instance of HttpMessage**
            if isinstance(raw_response, dict) and "body" in raw_response:
                response_body = raw_response["body"]
            elif hasattr(raw_response, "body"):
                response_body = raw_response.body
            else:
                self.context.logger.error(f"Unexpected response format: {raw_response}")
                return None

            # **Ensure response body is in JSON format**
            try:
                json_response = json.loads(response_body.decode("utf-8"))
            except json.JSONDecodeError as e:
                self.context.logger.error(f"Error decoding JSON response: {str(e)}")
                return None

            # **Log the actual JSON response**
            self.context.logger.info(f"Processed Response from Uniswap V3 API: {json.dumps(json_response, indent=2)}")

            # Return the complete data structure
            if "data" not in json_response:
                self.context.logger.error("No data field in response")
                return None

            return json_response["data"]

        except Exception as e:
            self.context.logger.error(f"Error fetching price for token {token_address}: {str(e)}")
            return None

    def get_llm_response(self, prompt: str) -> Generator[None, None, Optional[dict]]:
        """Get response from OpenAI API using ApiSpecs and return parsed JSON."""
        
        # Get the specs
        specs = self.openai_specs.get_spec()
        
        # Update the user message content with our prompt
        specs['parameters']['messages'][1]['content'] = prompt

        # Prepare the payload as JSON content
        content = json.dumps(specs['parameters']).encode('utf-8')

        # Make the API call with content as the body
        raw_response = yield from self.get_http_response(
            method=specs['method'],
            url=specs['url'],
            content=content, 
            headers=specs['headers']
        )

        try:
            # Parse the raw response body
            response_data = json.loads(raw_response.body)
            
            # Get the content from the first choice
            response_text = response_data.get('choices', [])[0].get('message', {}).get('content', '').strip()
            
            # Extract JSON from the markdown code block if present
            if '```json' in response_text:
                json_str = response_text.split('```json\n')[1].split('\n```')[0]
            else:
                json_str = response_text
                
            # Parse the JSON string into a dictionary
            decision_dict = json.loads(json_str)
            
            self.context.logger.info(f"Parsed decision: {decision_dict}")
            return decision_dict
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            self.context.logger.error(f"Error parsing OpenAI response: {e}")
            return None


    def get_next_event(self) -> Generator[None, None, Optional[Tuple[str,Dict[str, float]] ]]:
        """Get the next event: decide whether ot transact or not based on some data."""

        rebalancing_actions = yield from self.calculate_rebalancing_actions()


        self.context.logger.info("There should be some adjustment in the portfolio!")
        
        # # Generate and store the rebalancing report in IPFS
        # report_ipfs_hash = yield from self.generate_and_store_report(token_values, total_portfolio_value)
        # if report_ipfs_hash:
        #     self.context.logger.info(f"Rebalancing report stored in IPFS: https://gateway.autonolas.tech/ipfs/{report_ipfs_hash}")
        # else:
        #     self.context.logger.error("Failed to store rebalancing report in IPFS.")

        rebalancing_actions_json = json.dumps(rebalancing_actions, sort_keys=True) if rebalancing_actions else None
        return Event.TRANSACT.value, rebalancing_actions_json



        # Call the ledger connection (equivalent to web3.py)
        ledger_api_response = yield from self.get_ledger_api_response(
            performative=LedgerApiMessage.Performative.GET_STATE,
            ledger_callable="get_block_number",
            chain_id=ETHEREUM_CHAIN_ID,
        )

        # Check for errors on the response
        if ledger_api_response.performative != LedgerApiMessage.Performative.STATE:
            self.context.logger.error(
                f"Error while retrieving block number: {ledger_api_response}"
            )
            return None

        # Extract and return the block number
        block_number = cast(
            int, ledger_api_response.state.body["get_block_number_result"]
        )

        self.context.logger.error(f"Got block number: {block_number}")

        return block_number
    
    def calculate_rebalancing_actions(self) -> Generator[None, None, Optional[Dict[str, str]]]:
        """
        Calculate rebalancing actions based on current and target percentages.

        :return: Dictionary containing action and reason for rebalancing, or None if calculation fails.
        """
        # Start of method logging
        self.context.logger.info("Starting rebalancing calculation...")

        # Retrieve and log token values JSON
        token_values_json = self.synchronized_data.token_values
        self.context.logger.info(f"Token values JSON retrieved: {token_values_json}")

        token_values = {}
        # Convert JSON string back to a dictionary if it's not None
        if token_values_json is not None:
            try:
                token_values = json.loads(token_values_json)
                self.context.logger.info(f"Parsed token values dictionary: {token_values}")
            except json.JSONDecodeError as e:
                self.context.logger.error(f"Failed to decode token values JSON: {e}")
                token_values = {}
        else:
            self.context.logger.warning("Token values JSON is None. No tokens to rebalance.")
            return None

        # Retrieve and check total portfolio value
        total_portfolio_value = self.synchronized_data.total_portfolio_value
        if total_portfolio_value is None or total_portfolio_value <= 0:
            self.context.logger.error("Total portfolio value is None or zero; cannot calculate rebalancing.")
            return None

        token_data = yield from self.get_uniswap_token_price_specs()

        # Calculate current portfolio percentages
        weth_percentage = (token_values.get("WETH", 0) / total_portfolio_value) * 100
        usdc_percentage = (token_values.get("USDC", 0) / total_portfolio_value) * 100

        # Get the most recent data point (index 0) for both tokens
        weth_latest = token_data["WETH"][0]
        weth_yesterday = token_data["WETH"][1]
        usdc_latest = token_data["USDC"][0]
        usdc_yesterday = token_data["USDC"][1]

        # Create concise market summary
        market_summary = {
            "portfolio": {
                "total_value": f"${total_portfolio_value:,.2f}",
                "weth_percentage": f"{weth_percentage:.2f}%",
                "usdc_percentage": f"{usdc_percentage:.2f}%"
            },
            "market_data": {
                "weth": {
                    "current_price": f"${float(weth_latest['priceUSD']):.2f}",
                    "price_change": f"{((float(weth_latest['priceUSD']) - float(weth_yesterday['priceUSD'])) / float(weth_yesterday['priceUSD']) * 100):.2f}%",
                    "24h_volume": f"${float(weth_latest['volumeUSD']):,.2f}",
                    "volume_change": f"{((float(weth_latest['volumeUSD']) - float(weth_yesterday['volumeUSD'])) / float(weth_yesterday['volumeUSD']) * 100):.2f}%"
                }
            }
        }
        # Format market summary as string for prompt
        market_summary_str = json.dumps(market_summary, indent=2)

        # Construct prompt using market data
        prompt = f"""Based on the following portfolio and market data:
            {market_summary_str}

            Provide a single swap recommendation as JSON with two fields:
            1. 'action': specify direction (WETH to USDC or USDC to WETH) and percentage to swap (1-10%)
            2. 'reason': brief explanation in 10 words or less

            Response format example:
            {{
                "action": "swap 3% of weth to usdc",
                "reason": "decreasing volume suggests potential price decline"
        }}"""

        # Log LLM prompt
        self.context.logger.info(f"Generated LLM Prompt:\n{prompt}")

        # Send prompt to OpenAI API using get_llm_response
        rebalance_decision = yield from self.get_llm_response(prompt)

        # Log the AI's rebalance decision
        self.context.logger.info(f"OpenAI Response: {rebalance_decision}")

        return rebalance_decision

    def get_token_price_specs(self, symbol) -> Generator[None, None, Optional[float]]:
                """Get token price from Coingecko using ApiSpecs"""

                # Get the specs
                # specs = self.coingecko_specs.get_spec()
                specs = self.coinmarketcap_specs.get_spec()
                specs["parameters"]["symbol"] = symbol
                # Make the call
                raw_response = yield from self.get_http_response(**specs)

                # Process the response
                response = self.coinmarketcap_specs.process_response(raw_response)

                # Navigate to get the price
                token_data = response.get(symbol, {})
                price_info = token_data.get("quote", {}).get("USD", {})
                price = price_info.get("price", None)

                # Log and return the price
                self.context.logger.info(f"Got token price from CoinMarketCap: {price}")

                # Get the price
                # price = response.get("usd", None)
                # self.context.logger.info(f"Got token price from Coingecko: {price}")
                return price

    def generate_and_store_report(self, token_values: Dict[str, float], total_portfolio_value: float) -> Generator[None, None, Optional[str]]:
        """
        Generate the rebalancing report, store it in IPFS, and return the IPFS hash.

        :param token_values: Dictionary with tokens and their USD values.
        :param total_portfolio_value: Total value of the portfolio in USD.
        :return: IPFS hash of the stored report or None if storage fails.
        """
        from datetime import datetime

        # Generate the report JSON
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "variation_threshold": self.params.variation_threshold,
            "total_portfolio_value": total_portfolio_value,
            "tokens": []
        }

        for token, usd_value in token_values.items():
            target_percentage = self.params.target_percentages[self.params.tokens_to_rebalance.index(token)]
            current_percentage = (usd_value / total_portfolio_value) * 100
            token_price = yield from self.get_token_price_specs(token)
            current_token_amount = usd_value / token_price if token_price else 0

            report["tokens"].append({
                "token": token,
                "current_number_of_tokens": current_token_amount,
                "current_usd_value": usd_value,
                "current_percentage_in_portfolio": current_percentage,
                "target_percentage": target_percentage,
                "usd_deviation_from_target": current_percentage - target_percentage
            })

        # Store the report in IPFS
        report_ipfs_hash = yield from self.send_to_ipfs(
            filename="PortfolioRebalancer_Report.json", obj=report, filetype=SupportedFiletype.JSON
        )

        return report_ipfs_hash

class TxPreparationBehaviour(
    LearningBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """TxPreparationBehaviour"""

    matching_round: Type[AbstractRound] = TxPreparationRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            sender = self.context.agent_address

            adjustment_balances_json = self.synchronized_data.adjustment_balances
            self.context.logger.info(f"Adjustment balances JSON retrieved: {adjustment_balances_json}")

            # Get current token balances
            token_balances = yield from self.get_token_balances()
            if token_balances is None:
                self.context.logger.error("Failed to retrieve token balances.")
                return None

            # Log current token balances
            self.context.logger.info("Current token balances:")
            for token, balance in token_balances.items():
                self.context.logger.info(f"{token}: {balance}")

            # Parse adjustment action
            if adjustment_balances_json:
                adjustment_dict = json.loads(adjustment_balances_json)
                action = adjustment_dict.get('action', '')
                
                # Extract percentage and tokens from action string
                # Example: "swap 5% of weth to usdc"
                parts = action.split()
                try:
                    percentage = float(parts[1].replace('%', ''))
                    source_token = parts[3].upper() # Token after "of"
                    target_token = parts[5].upper() # Token after "to"
                    
                    # Calculate amount to swap based on percentage of source token balance
                    source_balance = token_balances.get(source_token)
                    if source_balance is None:
                        self.context.logger.error(f"No balance found for {source_token}")
                        return None
                        
                    amount_to_swap = source_balance * (percentage / 100)
                    
                    self.context.logger.info(f"Calculated swap amount:")
                    self.context.logger.info(f"Source token: {source_token}")
                    self.context.logger.info(f"Target token: {target_token}")
                    self.context.logger.info(f"Amount to swap: {amount_to_swap}")

                    # Create payload with source token, target token and amount
                    # Parse the swap payload
                    adjustment_balances = {
                        "source_token": source_token,
                        "target_token": target_token,
                        "amount": amount_to_swap,
                        "action": action,
                        "reason": "Rebalancing portfolio based on target allocation"
                    }

                except (IndexError, ValueError) as e:
                    self.context.logger.error(f"Failed to parse adjustment action: {e}")
                    return None

                # Get the transaction hash using the adjustment balances
                tx_hash = yield from self.generate_multisend_transactions(json.dumps(adjustment_balances))

            payload = TxPreparationPayload(
                sender=sender, tx_submitter=self.auto_behaviour_id(), tx_hash=tx_hash
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

     
    def get_token_balances(self) -> Generator[None, None, Optional[Dict[str, float]]]:
        """
        Get balances for each specified token from the deployed contract using parameters from self.params.

        :return: Dictionary of token balances, or None if an error occurs.
        """
        self.context.logger.info("Starting to fetch token balances for the portfolio.")

        # Retrieve portfolio address and tokens to rebalance from params
        portfolio_address = self.params.portfolio_address_string
        portfolio_manager_contract_address = self.params.portfolio_manager_contract_address_string

        # Use default token addresses and their decimals
        tokens_to_rebalance = {
            "USDC": {"address": USDC_ADDRESS, "decimals": 6},
            "WETH": {"address": WETH_ADDRESS, "decimals": 18}
        }

        # Log the portfolio details and tokens
        self.context.logger.info(f"Portfolio Address: {portfolio_address}")
        self.context.logger.info(f"Portfolio Manager Contract: {portfolio_manager_contract_address}")
        self.context.logger.info(f"Tokens to Rebalance: {tokens_to_rebalance}")

        # Prepare list of token addresses in the same order as the symbols
        token_addresses = [token_info["address"] for token_info in tokens_to_rebalance.values()]
        token_symbols = list(tokens_to_rebalance.keys())

        # Call the contract API to get all token balances at once
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,  # type: ignore
            contract_address=portfolio_manager_contract_address,
            contract_id=str(PORTFOLIOMANAGER.contract_id),
            contract_callable="get_user_balances",
            user=portfolio_address,
            tokens=token_addresses,
            chain_id=ETHEREUM_CHAIN_ID,
        )

        # Check if the response contains the expected balance data
        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(f"Error retrieving balances: {response_msg}")
            return None

        # Extract balances from the response
        balances_list = response_msg.raw_transaction.body.get("balances", None)
        if balances_list is None:
            self.context.logger.error("No balance data returned")
            return None

        # Create dictionary mapping token symbols to their balances
        balances = {}
        for i, (symbol, balance) in enumerate(zip(token_symbols, balances_list)):
            if balance is not None:
                # Convert the balance using the correct number of decimals for each token
                decimals = tokens_to_rebalance[symbol]["decimals"]
                readable_balance = float(balance) / (10 ** decimals)
                balances[symbol] = readable_balance
                self.context.logger.info(f"Balance for {symbol} (in readable format): {readable_balance}")
            else:
                self.context.logger.error(f"No balance data returned for {symbol}")
                balances[symbol] = None

        # Final printout of all token balances
        self.context.logger.info("Completed fetching balances for all tokens.")
        for token, balance in balances.items():
            if balance is not None:
                self.context.logger.info(f"Final balance for {token}: {balance}")
            else:
                self.context.logger.info(f"Balance for {token} could not be retrieved.")

        return balances if balances else None
    
    def get_adjust_balance_data(self, user: str, source_token: str, amount_to_swap: float, target_token: str) -> Generator[None, None, Dict]:
        """
        Get the transaction data for executing a rebalance through the PortfolioManager contract.

        :param user: Address of the user's portfolio
        :param source_token: Token to sell
        :param amount_to_swap: Amount of source token to swap
        :param target_token: Token to buy
        :return: Dictionary with transaction data
        """
        # Get contract addresses
        portfolio_manager_address = self.params.portfolio_manager_contract_address_string
        safe_address = self.params.safe_address

        # Define token addresses
        token_addresses = {
            "USDC": USDC_ADDRESS,
            "WETH": WETH_ADDRESS
        }

        # Define token decimals
        token_decimals = {
            "USDC": 6,
            "WETH": 18
        }

        # Convert human readable amount to token units with proper decimals
        source_decimals = token_decimals[source_token]
        amount_in_wei = int(amount_to_swap * (10 ** source_decimals))

        # Construct swap parameters
        swap_params = [{
            "tokenToSell": token_addresses[source_token],
            "tokenToBuy": token_addresses[target_token],
            "amountToSell": amount_in_wei,
            "amountOutMin": 0,  # Consider implementing slippage protection
            "poolFee": 3000  # Standard Uniswap v3 pool fee (0.3%)
        }]

        # Log the transaction parameters
        self.context.logger.info(f"Preparing executeRebalance transaction:")
        self.context.logger.info(f"Portfolio Manager: {portfolio_manager_address}")
        self.context.logger.info(f"User: {user}")
        self.context.logger.info(f"Swap params: {swap_params}")

        # Get the transaction data for executeRebalance
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=portfolio_manager_address,
            contract_id=str(PORTFOLIOMANAGER.contract_id),
            contract_callable="execute_rebalance",
            user=user,
            swaps=swap_params,
            chain_id=ETHEREUM_CHAIN_ID,
            from_address=safe_address,
        )

        # Check if transaction data was generated successfully
        if response_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error(f"Failed to prepare transaction data for executeRebalance: {response_msg}")
            return {}

        transaction_data_hex = response_msg.raw_transaction.body.get("data")
        if transaction_data_hex is None:
            self.context.logger.error("Transaction data is missing from response.")
            return {}

        # Return transaction data
        transaction_data = {
            "to_address": portfolio_manager_address,
            "data": bytes.fromhex(transaction_data_hex[2:])  # Convert hex string to bytes without "0x"
        }
        self.context.logger.info(f"Prepared executeRebalance transaction data: {transaction_data}")
        
        return transaction_data
    
    def generate_multisend_transactions(self, adjustment_balances_json: str) -> Generator[None, None, Optional[str]]:
        """Generate multisend transactions for each token adjustment balance."""

        # Parse the adjustment balances JSON to get the target balances for each token
        multi_send_txs = []
        adjustment_balances = json.loads(adjustment_balances_json)
        portfolio_address = self.params.portfolio_address_string
        
        # Log the rebalancing action and reason
        self.context.logger.info(f"Rebalancing action: {adjustment_balances['action']}")
        self.context.logger.info(f"Rebalancing reason: {adjustment_balances['reason']}")

        # Get balance adjustment transaction data
        balance_adjustment_data = yield from self.get_adjust_balance_data(
            user=portfolio_address,
            source_token=adjustment_balances["source_token"],
            amount_to_swap=adjustment_balances["amount"],
            target_token=adjustment_balances["target_token"]
        )

        if not balance_adjustment_data:
            self.context.logger.error("Failed to prepare balance adjustment transaction")
            return None

        multi_send_txs.append({
            "operation": MultiSendOperation.CALL,
            "to": balance_adjustment_data["to_address"],
            "data": balance_adjustment_data["data"],
            "value": ZERO_VALUE,
        })
        self.context.logger.info(f"Prepared balance adjustment data: {balance_adjustment_data}")


        # Step 3: Pack the multisend transactions into a single call
        self.context.logger.info(f"Preparing multisend transaction with txs: {multi_send_txs}")
        contract_api_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_RAW_TRANSACTION,
            contract_address=self.params.multisend_address,
            contract_id=str(MultiSendContract.contract_id),
            contract_callable="get_tx_data",
            multi_send_txs=multi_send_txs,
            chain_id=ETHEREUM_CHAIN_ID,
        )

        # Step 4: Check for errors and prepare Safe transaction hash
        if contract_api_msg.performative != ContractApiMessage.Performative.RAW_TRANSACTION:
            self.context.logger.error("Could not get Multisend tx hash.")
            return None

        multisend_data = contract_api_msg.raw_transaction.body["data"]
        # Strip "0x" if it exists, then convert
        multisend_data = multisend_data[2:] if multisend_data.startswith("0x") else multisend_data
        data_bytes = bytes.fromhex(multisend_data)

        safe_tx_hash = yield from self._build_safe_tx_hash(
            to_address=self.params.multisend_address,
            value=ZERO_VALUE,
            data=data_bytes,
            operation=SafeOperation.DELEGATE_CALL.value,
        )
        if safe_tx_hash is None:
            self.context.logger.error("Failed to prepare Safe transaction hash.")
        else:
            self.context.logger.info(f"Safe transaction hash successfully prepared: {safe_tx_hash}")

        return safe_tx_hash

    def _build_safe_tx_hash(
        self,
        to_address: str,
        value: int = ZERO_VALUE,
        data: bytes = EMPTY_CALL_DATA,
        operation: int = SafeOperation.CALL.value,
    ) -> Generator[None, None, Optional[str]]:
        """Prepares and returns the safe tx hash for a multisend tx."""

        self.context.logger.info(
            f"Preparing Safe transaction [{self.synchronized_data.safe_contract_address}]"
        )

        # Prepare the safe transaction
        response_msg = yield from self.get_contract_api_response(
            performative=ContractApiMessage.Performative.GET_STATE,  # type: ignore
            contract_address=self.synchronized_data.safe_contract_address,
            contract_id=str(GnosisSafeContract.contract_id),
            contract_callable="get_raw_safe_transaction_hash",
            to_address=to_address,
            value=value,
            data=data,
            safe_tx_gas=SAFE_GAS,
            chain_id=ETHEREUM_CHAIN_ID,
            operation=operation,
        )

        # Check for errors
        if response_msg.performative != ContractApiMessage.Performative.STATE:
            self.context.logger.error(
                "Couldn't get safe tx hash. Expected response performative "
                f"{ContractApiMessage.Performative.STATE.value!r}, "  # type: ignore
                f"received {response_msg.performative.value!r}: {response_msg}."
            )
            return None

        # Extract the hash and check it has the correct length
        tx_hash: Optional[str] = response_msg.state.body.get("tx_hash", None)

        if tx_hash is None or len(tx_hash) != TX_HASH_LENGTH:
            self.context.logger.error(
                "Something went wrong while trying to get the safe transaction hash. "
                f"Invalid hash {tx_hash!r} was returned."
            )
            return None

        # Transaction to hex
        tx_hash = tx_hash[2:]  # strip the 0x

        safe_tx_hash = hash_payload_to_hex(
            safe_tx_hash=tx_hash,
            ether_value=value,
            safe_tx_gas=SAFE_GAS,
            to_address=to_address,
            data=data,
            operation=operation,
        )

        self.context.logger.info(f"Safe transaction hash is {safe_tx_hash}")

        return safe_tx_hash

class LearningRoundBehaviour(AbstractRoundBehaviour):
    """LearningRoundBehaviour"""

    initial_behaviour_cls = ApiSelectionBehaviour
    abci_app_cls = LearningAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [  # type: ignore
        ApiSelectionBehaviour,
        DataPullBehaviour,
        AlternativeDataPullBehaviour,
        DecisionMakingBehaviour,
        TxPreparationBehaviour,

    ]

