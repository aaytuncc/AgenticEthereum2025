#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2024 Valory AG
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


"""Updates fetched agent with correct config"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

import json

def main() -> None:
    """Main"""
    load_dotenv()

    with open(Path("learning_agent", "aea-config.yaml"), "r", encoding="utf-8") as file:
        config = list(yaml.safe_load_all(file))

        # Ledger RPCs
        if os.getenv("GNOSIS_LEDGER_RPC"):
            config[2]["config"]["ledger_apis"]["gnosis"][
                "address"
            ] = f"${{str:{os.getenv('GNOSIS_LEDGER_RPC')}}}"
        
        if os.getenv("ETHEREUM_LEDGER_RPC"):
            config[2]["config"]["ledger_apis"]["ethereum"][
                "address"
            ] = f"${{str:{os.getenv('ETHEREUM_LEDGER_RPC')}}}"

        # Params
        if os.getenv("COINGECKO_API_KEY"):
            # Coingecko API key (params)
            config[-1]["models"]["params"]["args"][
                "coingecko_api_key"
            ] = f"${{str:{os.getenv('COINGECKO_API_KEY')}}}"  # type: ignore

            # Coingecko API key (ApiSpecs)
            config[-1]["models"]["coingecko_specs"]["args"]["parameters"][
                "x_cg_demo_api_key"
            ] = f"${{str:{os.getenv('COINGECKO_API_KEY')}}}"  # type: ignore

            # CoinMarketCap API key (ApiSpecs)
            config[-1]["models"]["coinmarketcap_specs"]["args"]["parameters"][
                "CMC_PRO_API_KEY"
            ] = f"${{str:{os.getenv('COINMARKETCAP_API_KEY')}}}"  # type: ignore

            # Graph API key (ApiSpecs)
            config[-1]["models"]["thegraph_specs"]["args"]["url"] = (
                f"https://gateway.thegraph.com/api/{os.getenv('THEGRAPH_API_KEY', '')}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
            )

            # Arli AI API key (ApiSpecs)
            config[-1]["models"]["arli_specs"]["args"]["headers"][
                "Authorization"
            ] = f"Bearer {os.getenv('ARLI_API_KEY')}"  # type: ignore

            # OpenAI AI API key (ApiSpecs)
            config[-1]["models"]["openai_specs"]["args"]["headers"][
                "Authorization"
            ] = f"Bearer {os.getenv('OPENAI_API_KEY')}"  # type: ignore

            # ALL_PARTICIPANTS
            config[-1]["models"]["params"]["args"]["setup"][
                "all_participants"
            ] = f"${{list:{os.getenv('ALL_PARTICIPANTS')}}}"  # type: ignore

            # SAFE_CONTRACT_ADDRESS_SINGLE
            config[-1]["models"]["params"]["args"]["setup"][
                "safe_contract_address"
            ] = f"${{str:{os.getenv('SAFE_CONTRACT_ADDRESS_SINGLE')}}}"  # type: ignore

            # TRANSFER_TARGET_ADDRESS
            config[-1]["models"]["params"]["args"][
                "transfer_target_address"
            ] = f"${{str:{os.getenv('TRANSFER_TARGET_ADDRESS')}}}"  # type: ignore

            # MOCK_CONTRACT_ADDRESS
            config[-1]["models"]["params"]["args"][
                "mock_contract_address"
            ] = f"${{str:{os.getenv('MOCK_CONTRACT_ADDRESS')}}}"  # type: ignore

            # MOCK_CONTRACT_ADDRESS
            config[-1]["models"]["params"]["args"][
                "portfolio_manager_contract_address"
            ] = f"${{str:{os.getenv('PORTFOLIO_MANAGER_CONTRACT_ADDRESS')}}}"  # type: ignore

            # PORTFOLIO_ADDRESS
            config[-1]["models"]["params"]["args"][
                "portfolio_address"
            ] = f"${{str:{os.getenv('PORTFOLIO_ADDRESS')}}}"  # type: ignore

            # API_SELECTION
            config[-1]["models"]["params"]["args"][
                "api_selection"
            ] = f"${{str:{os.getenv('API_SELECTION')}}}"  # type: ignore

            # TOKENS_TO_REBALANCE
            config[-1]["models"]["params"]["args"]["tokens_to_rebalance"] = [
                f"${{str:{token}}}" for token in json.loads(os.getenv("TOKENS_TO_REBALANCE"))
                ]   # type: ignore
            
            #TARGET_PERCENTAGES
            config[-1]["models"]["params"]["args"]["target_percentages"] = [
                f"${{float:{float(percent)}}}" for percent in json.loads(os.getenv("TARGET_PERCENTAGES"))
            ]  # type: ignore

            # VARIATION_THRESHOLD
            config[-1]["models"]["params"]["args"][
                "variation_threshold"
            ] = f"${{float:{os.getenv('VARIATION_THRESHOLD', '3.0')}}}"  # type: ignore
    with open(Path("learning_agent", "aea-config.yaml"), "w", encoding="utf-8") as file:
        yaml.dump_all(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
