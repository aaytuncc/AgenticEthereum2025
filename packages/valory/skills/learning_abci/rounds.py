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

"""This package contains the rounds of LearningAbciApp."""

from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple

from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name,
)
from packages.valory.skills.learning_abci.payloads import (
    ApiSelectionPayload,
    DataPullPayload,
    AlternativeDataPullPayload,
    DecisionMakingPayload,
    TxPreparationPayload,
)


class Event(Enum):
    """LearningAbciApp Events"""

    DONE = "done"
    ERROR = "error"
    TRANSACT = "transact"
    NO_MAJORITY = "no_majority"
    ROUND_TIMEOUT = "round_timeout"
    COINGECKO = "coingecko"
    COINMARKETCAP = "coinmarketcap"




class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application, so all the agents share the same data.
    """

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)

    @property
    def token_values(self) -> Optional[str]:
        """Get the token values."""
        return self.db.get("token_values", None)

    @property
    def total_portfolio_value(self) -> Optional[float]:
        """Get the total portfolio value."""
        return self.db.get("total_portfolio_value", None)

    @property
    def adjustment_balances(self) -> Optional[str]:
        """Get the total adjsutment balances."""
        return self.db.get("adjustment_balances", None)

    @property
    def participant_to_data_round(self) -> DeserializedCollection:
        """Agent to payload mapping for the DataPullRound."""
        return self._get_deserialized("participant_to_data_round")

    @property
    def participant_to_decision_making_round(self) -> DeserializedCollection:
        """Agent to payload mapping for the DecisionMakingRound."""
        return self._get_deserialized("participant_to_decision_making_round")

    @property
    def api_selection(self) -> str:
        """Get the api selection choice."""
        return self.db.get("api_selection", "coingecko")

    @property
    def most_voted_tx_hash(self) -> Optional[float]:
        """Get the token most_voted_tx_hash."""
        return self.db.get("most_voted_tx_hash", None)

    @property
    def participant_to_tx_round(self) -> DeserializedCollection:
        """Get the participants to the tx round."""
        return self._get_deserialized("participant_to_tx_round")

    @property
    def tx_submitter(self) -> str:
        """Get the round that submitted a tx to transaction_settlement_abci."""
        return str(self.db.get_strict("tx_submitter"))


class ApiSelectionRound(CollectSameUntilThresholdRound):
    """ApiSelectionRound: decides which API to use (CoinGecko or CoinMarketCap)."""

    payload_class = ApiSelectionPayload
    synchronized_data_class = SynchronizedData
    # required_class_attributes = ()


    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """
        Determine the outcome based on the payloads collected from agents.
        """
        if self.threshold_reached:
            if self.synchronized_data.api_selection != self.most_voted_payload:
                updated_synchronized_data = self.synchronized_data.update(
                    api_selection=self.most_voted_payload,
                    synchronized_data_class=SynchronizedData
                )
                return updated_synchronized_data, Event.COINMARKETCAP
            return self.synchronized_data, Event.COINGECKO
        return None



class AlternativeDataPullRound(CollectSameUntilThresholdRound):
    """DataPullRound"""

    payload_class = AlternativeDataPullPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    # required_class_attributes = ()


    # Collection key specifies where in the synchronized data the agento to payload mapping will be stored
    collection_key = get_name(SynchronizedData.participant_to_data_round)

    # Selection key specifies how to extract all the different objects from each agent's payload
    # and where to store it in the synchronized data. Notice that the order follows the same order
    # from the payload class.
    selection_key = (
        get_name(SynchronizedData.token_values),
        get_name(SynchronizedData.total_portfolio_value),
    )

    # Event.ROUND_TIMEOUT  # this needs to be referenced for static checkers

class DataPullRound(CollectSameUntilThresholdRound):
    """DataPullRound"""

    payload_class = DataPullPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    # required_class_attributes = ()


    # Collection key specifies where in the synchronized data the agento to payload mapping will be stored
    collection_key = get_name(SynchronizedData.participant_to_data_round)

    # Selection key specifies how to extract all the different objects from each agent's payload
    # and where to store it in the synchronized data. Notice that the order follows the same order
    # from the payload class.
    selection_key = (
        get_name(SynchronizedData.token_values),
        get_name(SynchronizedData.total_portfolio_value),

    )

    # Event.ROUND_TIMEOUT  # this needs to be referenced for static checkers

class DecisionMakingRound(CollectSameUntilThresholdRound):
    """DecisionMakingRound
        If the threshold is reached, retrieves and updates adjustment_balances in synchronized_data and triggers a TRANSACT event.
        Returns NO_MAJORITY if a consensus cannot be reached or ERROR if data is missing. Returns None if voting continues.
    """

    payload_class = DecisionMakingPayload
    synchronized_data_class = SynchronizedData
    # required_class_attributes = ()


    # Define collection and selection keys
    collection_key = get_name(SynchronizedData.participant_to_decision_making_round)
    selection_key = (
        get_name(SynchronizedData.adjustment_balances),
    )

    def end_block(self) -> Optional[Tuple[BaseSynchronizedData, Event]]:
        """Process the end of the block."""

        if self.threshold_reached:
            # Search for the payload matching the most voted event
            most_voted_payload_data = None
            for payload in self.collection.values():
                if payload.event == self.most_voted_payload:
                    most_voted_payload_data = payload
                    break

            if most_voted_payload_data is None:
                self.context.logger.error("Most voted payload data not found.")
                return self.synchronized_data, Event.ERROR

            # Extract `adjustment_balances` and update synchronized data
            adjustment_balances = most_voted_payload_data.adjustment_balances
            if adjustment_balances is not None:
                new_synchronized_data = self.synchronized_data.update(
                    adjustment_balances=adjustment_balances
                )
            else:
                self.context.logger.warning("Adjustment balances not found in payload.")
                return self.synchronized_data, Event.DONE

            return new_synchronized_data, Event.TRANSACT

        if not self.is_majority_possible(
            self.collection, self.synchronized_data.nb_participants
        ):
            return self.synchronized_data, Event.NO_MAJORITY

        return None

class TxPreparationRound(CollectSameUntilThresholdRound):
    """TxPreparationRound"""

    payload_class = TxPreparationPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    # required_class_attributes = ()

    collection_key = get_name(SynchronizedData.participant_to_tx_round)
    selection_key = (
        get_name(SynchronizedData.tx_submitter),
        get_name(SynchronizedData.most_voted_tx_hash),
    )

    # Event.ROUND_TIMEOUT  # this needs to be referenced for static checkers


class FinishedDecisionMakingRound(DegenerateRound):
    """FinishedDecisionMakingRound"""


class FinishedTxPreparationRound(DegenerateRound):
    """FinishedLearningRound"""


class LearningAbciApp(AbciApp[Event]):
    """LearningAbciApp"""

    initial_round_cls: AppState = ApiSelectionRound
    initial_states: Set[AppState] = {
        ApiSelectionRound,
    }
    transition_function: AbciAppTransitionFunction = {
        ApiSelectionRound: {
            Event.NO_MAJORITY: ApiSelectionRound,
            Event.ROUND_TIMEOUT: ApiSelectionRound,
            Event.COINGECKO: DataPullRound,
            Event.COINMARKETCAP: AlternativeDataPullRound,
        },
        DataPullRound: {
            Event.NO_MAJORITY: DataPullRound,
            Event.ROUND_TIMEOUT: DataPullRound,
            Event.DONE: DecisionMakingRound,
        },
        AlternativeDataPullRound: {
            Event.NO_MAJORITY: AlternativeDataPullRound,
            Event.ROUND_TIMEOUT: AlternativeDataPullRound,
            Event.DONE: DecisionMakingRound,
        },
        DecisionMakingRound: {
            Event.NO_MAJORITY: DecisionMakingRound,
            Event.ROUND_TIMEOUT: DecisionMakingRound,
            Event.DONE: FinishedDecisionMakingRound,
            Event.ERROR: FinishedDecisionMakingRound,
            Event.TRANSACT: TxPreparationRound,
        },
        TxPreparationRound: {
            Event.NO_MAJORITY: TxPreparationRound,
            Event.ROUND_TIMEOUT: TxPreparationRound,
            Event.DONE: FinishedTxPreparationRound,
        },
        FinishedDecisionMakingRound: {},
        FinishedTxPreparationRound: {},
    }
    final_states: Set[AppState] = {
        FinishedDecisionMakingRound,
        FinishedTxPreparationRound,
    }
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        ApiSelectionRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedDecisionMakingRound: set(),
        FinishedTxPreparationRound: {get_name(SynchronizedData.most_voted_tx_hash)},
    }
