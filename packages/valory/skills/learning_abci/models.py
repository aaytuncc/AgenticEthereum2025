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

"""This module contains the shared state for the abci skill of LearningAbciApp."""

from typing import Any

from packages.valory.skills.abstract_round_abci.models import ApiSpecs, BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.learning_abci.rounds import LearningAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = LearningAbciApp


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.coingecko_price_template = self._ensure(
            "coingecko_price_template", kwargs, str
        )
        self.coingecko_api_key = kwargs.get("coingecko_api_key", None)
        self.coinmarketcap_api_key = kwargs.get("coinmarketcap_api_key", None)
        
        self.thegraph_api_key = kwargs.get("thegraph_api_key", None)
        self.openai_api_key = kwargs.get("openai_api_key", None)

        
        self.arli_api_key = kwargs.get("arli_api_key", None)

        self.api_selection_string: str = self._ensure("api_selection", kwargs, str)

        self.transfer_target_address = self._ensure(
            "transfer_target_address", kwargs, str
        )
        self.olas_token_address = self._ensure("olas_token_address", kwargs, str)

        # multisend address is used in other skills, so we cannot pop it using _ensure
        self.multisend_address = kwargs.get("multisend_address", None)

        # Rebalancing settings
        self.tokens_to_rebalance: List[str] = self._ensure("tokens_to_rebalance", kwargs, list)
        self.target_percentages: List[float] = self._ensure("target_percentages", kwargs, list)
        self.variation_threshold: float = self._ensure("variation_threshold", kwargs, float)
        self.portfolio_address_string: str = self._ensure("portfolio_address", kwargs, str)
        self.mock_contract_address_string: str = self._ensure("mock_contract_address", kwargs, str)
        self.portfolio_manager_contract_address_string: str = self._ensure("portfolio_manager_contract_address", kwargs, str)

        
        #Neeed for from field while interacting with protected contract of MockTrade.
        self.safe_address: str = kwargs.get("setup", {}).get("safe_contract_address", "")



        super().__init__(*args, **kwargs)

    def validate_params(self) -> None:
        """Validate that the token rebalancing parameters are set correctly."""
        if len(self.tokens_to_rebalance) != len(self.target_percentages):
            raise ValueError("The number of tokens must match the number of target percentages.")
        if sum(self.target_percentages) != 100:
            raise ValueError("Target percentages must sum to 100.")
        if not 0 <= self.variation_threshold <= 100:
            raise ValueError("Variation threshold must be between 0 and 100.")


class CoingeckoSpecs(ApiSpecs):
    """A model that wraps ApiSpecs for Coingecko API."""

class CoinMarketCapSpecs(ApiSpecs):
    """A model that wraps ApiSpecs for CoinMarketCap API."""

class TheGraphSpecs(ApiSpecs):
    """A model that wraps ApiSpecs for TheGraph API."""

class ArliSpecs(ApiSpecs):
    """A model that wraps ApiSpecs for Coinmarketcap API."""

class OpenAISpecs(ApiSpecs):
    """A model that wraps ApiSpecs for Coinmarketcap API."""