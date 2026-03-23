"""
Comprehensive unit tests for detect_anomalies.py module.

All tests use synthetic/inline data and do not depend on external CSV files.
"""

import io
import pytest
from unittest.mock import patch, mock_open
from collections import defaultdict

import detect_anomalies as da


# ---------------------------------------------------------------------------
# Helper: factory for transaction dicts
# ---------------------------------------------------------------------------

def make_txn(
    row=2,
    step=1,
    txn_type="TRANSFER",
    amount=1000.0,
    name_orig="C_orig",
    old_balance_org=5000.0,
    new_balance_orig=4000.0,
    name_dest="C_dest",
    old_balance_dest=0.0,
    new_balance_dest=1000.0,
    is_fraud=0,
    is_flagged_fraud=0,
):
    return {
        "row": row,
        "step": step,
        "type": txn_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged_fraud,
    }


# ===================================================================
# Tests for load_transactions
# ===================================================================

class TestLoadTransactions:
    """Tests for load_transactions(filepath)."""

    def _csv_content(self, rows):
        """Build CSV string from list of row-dicts."""
        header = (
            "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
            "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
        )
        lines = [header]
        for r in rows:
            lines.append(
                f"{r['step']},{r['type']},{r['amount']},{r['nameOrig']},"
                f"{r['oldbalanceOrg']},{r['newbalanceOrig']},"
                f"{r['nameDest']},{r['oldbalanceDest']},{r['newbalanceDest']},"
                f"{r['isFraud']},{r['isFlaggedFraud']}"
            )
        return "\n".join(lines) + "\n"

    def test_basic_load(self):
        """Loads a valid CSV and returns correct transaction dicts."""
        csv_rows = [
            {
                "step": "1", "type": "TRANSFER", "amount": "500.0",
                "nameOrig": "C100", "oldbalanceOrg": "1000.0",
                "newbalanceOrig": "500.0", "nameDest": "C200",
                "oldbalanceDest": "0.0", "newbalanceDest": "500.0",
                "isFraud": "0", "isFlaggedFraud": "0",
            }
        ]
        content = self._csv_content(csv_rows)
        with patch("builtins.open", mock_open(read_data=content)):
            txns = da.load_transactions("fake.csv")

        assert len(txns) == 1
        t = txns[0]
        assert t["row"] == 2
        assert t["step"] == 1
        assert t["type"] == "TRANSFER"
        assert t["amount"] == 500.0
        assert t["nameOrig"] == "C100"
        assert t["oldbalanceOrg"] == 1000.0
        assert t["newbalanceOrig"] == 500.0
        assert t["nameDest"] == "C200"
        assert t["oldbalanceDest"] == 0.0
        assert t["newbalanceDest"] == 500.0
        assert t["isFraud"] == 0
        assert t["isFlaggedFraud"] == 0

    def test_skips_empty_name_orig(self):
        """Rows with empty nameOrig are skipped."""
        csv_rows = [
            {
                "step": "1", "type": "TRANSFER", "amount": "100",
                "nameOrig": "", "oldbalanceOrg": "0",
                "newbalanceOrig": "0", "nameDest": "C200",
                "oldbalanceDest": "0", "newbalanceDest": "100",
                "isFraud": "0", "isFlaggedFraud": "0",
            },
            {
                "step": "2", "type": "CASH_OUT", "amount": "200",
                "nameOrig": "C300", "oldbalanceOrg": "200",
                "newbalanceOrig": "0", "nameDest": "C400",
                "oldbalanceDest": "0", "newbalanceDest": "200",
                "isFraud": "0", "isFlaggedFraud": "0",
            },
        ]
        content = self._csv_content(csv_rows)
        with patch("builtins.open", mock_open(read_data=content)):
            txns = da.load_transactions("fake.csv")

        assert len(txns) == 1
        assert txns[0]["nameOrig"] == "C300"

    def test_handles_scientific_notation_in_new_balance_dest(self):
        """newbalanceDest with 'E' notation is converted correctly."""
        csv_rows = [
            {
                "step": "1", "type": "TRANSFER", "amount": "100",
                "nameOrig": "C1", "oldbalanceOrg": "1000",
                "newbalanceOrig": "900", "nameDest": "C2",
                "oldbalanceDest": "0", "newbalanceDest": "1.5E6",
                "isFraud": "0", "isFlaggedFraud": "0",
            }
        ]
        content = self._csv_content(csv_rows)
        with patch("builtins.open", mock_open(read_data=content)):
            txns = da.load_transactions("fake.csv")

        assert txns[0]["newbalanceDest"] == 1.5e6

    def test_row_numbering(self):
        """Row numbers start at 2 and increment, skipping filtered rows."""
        csv_rows = [
            {
                "step": "1", "type": "TRANSFER", "amount": "100",
                "nameOrig": "C1", "oldbalanceOrg": "100",
                "newbalanceOrig": "0", "nameDest": "C2",
                "oldbalanceDest": "0", "newbalanceDest": "100",
                "isFraud": "0", "isFlaggedFraud": "0",
            },
            {
                "step": "2", "type": "TRANSFER", "amount": "200",
                "nameOrig": "", "oldbalanceOrg": "200",
                "newbalanceOrig": "0", "nameDest": "C3",
                "oldbalanceDest": "0", "newbalanceDest": "200",
                "isFraud": "0", "isFlaggedFraud": "0",
            },
            {
                "step": "3", "type": "CASH_OUT", "amount": "300",
                "nameOrig": "C4", "oldbalanceOrg": "300",
                "newbalanceOrig": "0", "nameDest": "C5",
                "oldbalanceDest": "0", "newbalanceDest": "300",
                "isFraud": "0", "isFlaggedFraud": "0",
            },
        ]
        content = self._csv_content(csv_rows)
        with patch("builtins.open", mock_open(read_data=content)):
            txns = da.load_transactions("fake.csv")

        assert len(txns) == 2
        assert txns[0]["row"] == 2  # idx=0 -> 0+2=2
        assert txns[1]["row"] == 4  # idx=2 -> 2+2=4


# ===================================================================
# Tests for build_balance_chains
# ===================================================================

class TestBuildBalanceChains:
    """Tests for build_balance_chains(transactions)."""

    def test_empty_transactions(self):
        """Empty input returns no chains."""
        assert da.build_balance_chains([]) == []

    def test_single_transaction_no_chain(self):
        """A single transaction cannot form a chain (minimum 2)."""
        txns = [make_txn(old_balance_org=100.0, new_balance_orig=50.0)]
        chains = da.build_balance_chains(txns)
        assert chains == []

    def test_two_linked_transactions(self):
        """Two transactions linked by balance continuity form a chain."""
        txns = [
            make_txn(row=2, old_balance_org=1000.0, new_balance_orig=500.0),
            make_txn(row=3, old_balance_org=500.0, new_balance_orig=200.0),
        ]
        chains = da.build_balance_chains(txns)
        assert len(chains) == 1
        assert len(chains[0]) == 2
        assert chains[0][0]["row"] == 2
        assert chains[0][1]["row"] == 3

    def test_three_linked_transactions(self):
        """Three transactions chain together."""
        txns = [
            make_txn(row=2, old_balance_org=1000.0, new_balance_orig=800.0),
            make_txn(row=3, old_balance_org=800.0, new_balance_orig=500.0),
            make_txn(row=4, old_balance_org=500.0, new_balance_orig=100.0),
        ]
        chains = da.build_balance_chains(txns)
        assert len(chains) == 1
        assert len(chains[0]) == 3

    def test_zero_balance_not_indexed(self):
        """Transactions with oldbalanceOrg==0 are NOT indexed in by_old_balance,
        so they cannot be linked as a successor."""
        txns = [
            make_txn(row=2, old_balance_org=1000.0, new_balance_orig=0.0),
            make_txn(row=3, old_balance_org=0.0, new_balance_orig=500.0),
        ]
        chains = da.build_balance_chains(txns)
        assert chains == []

    def test_new_balance_zero_breaks_chain(self):
        """If newbalanceOrig <= 0, chain building stops."""
        txns = [
            make_txn(row=2, old_balance_org=1000.0, new_balance_orig=0.0),
            make_txn(row=3, old_balance_org=1000.0, new_balance_orig=500.0),
        ]
        chains = da.build_balance_chains(txns)
        # Neither can form a chain of length >= 2
        assert chains == []

    def test_used_set_prevents_reuse(self):
        """Transactions already used in one chain cannot appear in another."""
        txns = [
            make_txn(row=2, old_balance_org=1000.0, new_balance_orig=500.0),
            make_txn(row=3, old_balance_org=500.0, new_balance_orig=200.0),
            # This txn also has oldbalanceOrg=500 but row 3 is already used
            make_txn(row=4, old_balance_org=500.0, new_balance_orig=100.0),
        ]
        chains = da.build_balance_chains(txns)
        # Only one chain of 2 (rows 2,3); row 4 can't link
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_two_independent_chains(self):
        """Two separate chains formed from independent balance continuities."""
        txns = [
            # Chain A
            make_txn(row=2, old_balance_org=1000.0, new_balance_orig=700.0),
            make_txn(row=3, old_balance_org=700.0, new_balance_orig=400.0),
            # Chain B
            make_txn(row=4, old_balance_org=2000.0, new_balance_orig=1500.0),
            make_txn(row=5, old_balance_org=1500.0, new_balance_orig=1000.0),
        ]
        chains = da.build_balance_chains(txns)
        assert len(chains) == 2


# ===================================================================
# Tests for build_destination_map
# ===================================================================

class TestBuildDestinationMap:
    """Tests for build_destination_map(transactions)."""

    def test_empty_transactions(self):
        assert da.build_destination_map([]) == {}

    def test_groups_by_destination(self):
        txns = [
            make_txn(name_dest="D1"),
            make_txn(name_dest="D2"),
            make_txn(name_dest="D1"),
        ]
        dm = da.build_destination_map(txns)
        assert len(dm["D1"]) == 2
        assert len(dm["D2"]) == 1


# ===================================================================
# Tests for detect_repeated_high_value
# ===================================================================

class TestDetectRepeatedHighValue:
    """Tests for detect_repeated_high_value(chain)."""

    def test_no_high_value(self):
        """All amounts <= threshold -> no anomalies."""
        chain = [
            make_txn(amount=5000.0),
            make_txn(amount=10000.0),  # exactly 10000 -> NOT high value
        ]
        assert da.detect_repeated_high_value(chain) == []

    def test_boundary_exactly_10000_not_triggered(self):
        """Amount of exactly 10000 should NOT trigger (threshold is >10000)."""
        chain = [
            make_txn(amount=10000.0),
            make_txn(amount=10000.0),
        ]
        assert da.detect_repeated_high_value(chain) == []

    def test_boundary_10001_triggers(self):
        """Two consecutive amounts of 10001 should trigger."""
        chain = [
            make_txn(amount=10001.0),
            make_txn(amount=10001.0),
        ]
        anomalies = da.detect_repeated_high_value(chain)
        assert len(anomalies) == 1
        assert anomalies[0]["pattern"] == "repeated_high_value"
        assert anomalies[0]["total_amount"] == 20002.0

    def test_single_high_value_no_anomaly(self):
        """One high-value txn in chain does not trigger (need >= 2 consecutive)."""
        chain = [
            make_txn(amount=50000.0),
            make_txn(amount=100.0),
        ]
        assert da.detect_repeated_high_value(chain) == []

    def test_two_consecutive_high_value(self):
        """Two consecutive high-value transactions flagged."""
        chain = [
            make_txn(row=2, amount=15000.0),
            make_txn(row=3, amount=20000.0),
        ]
        anomalies = da.detect_repeated_high_value(chain)
        assert len(anomalies) == 1
        assert anomalies[0]["pattern"] == "repeated_high_value"
        assert len(anomalies[0]["transactions"]) == 2
        assert anomalies[0]["total_amount"] == 35000.0

    def test_three_consecutive_high_value(self):
        """Three consecutive high-value transactions form one group."""
        chain = [
            make_txn(amount=15000.0),
            make_txn(amount=20000.0),
            make_txn(amount=25000.0),
        ]
        anomalies = da.detect_repeated_high_value(chain)
        assert len(anomalies) == 1
        assert len(anomalies[0]["transactions"]) == 3
        assert anomalies[0]["total_amount"] == 60000.0

    def test_broken_by_low_value(self):
        """High-value group broken by a low-value transaction."""
        chain = [
            make_txn(amount=15000.0),
            make_txn(amount=20000.0),
            make_txn(amount=500.0),  # breaks the group
            make_txn(amount=30000.0),
            make_txn(amount=40000.0),
        ]
        anomalies = da.detect_repeated_high_value(chain)
        assert len(anomalies) == 2

    def test_high_value_at_end_of_chain(self):
        """Consecutive high values at the end of a chain are detected."""
        chain = [
            make_txn(amount=100.0),
            make_txn(amount=15000.0),
            make_txn(amount=20000.0),
        ]
        anomalies = da.detect_repeated_high_value(chain)
        assert len(anomalies) == 1


# ===================================================================
# Tests for detect_transfer_then_cashout
# ===================================================================

class TestDetectTransferThenCashout:
    """Tests for detect_transfer_then_cashout(chain)."""

    def test_transfer_then_cashout(self):
        """TRANSFER -> CASH_OUT detected."""
        chain = [
            make_txn(txn_type="TRANSFER"),
            make_txn(txn_type="CASH_OUT"),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        patterns = [a["pattern"] for a in anomalies]
        assert "transfer_then_cashout" in patterns

    def test_transfer_transfer_cashout(self):
        """TRANSFER -> TRANSFER -> CASH_OUT detected."""
        chain = [
            make_txn(txn_type="TRANSFER", amount=100.0),
            make_txn(txn_type="TRANSFER", amount=200.0),
            make_txn(txn_type="CASH_OUT", amount=300.0),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        patterns = [a["pattern"] for a in anomalies]
        assert "transfer_transfer_cashout" in patterns
        # The first TRANSFER -> CASH_OUT pair is NOT detected because
        # chain[0] is TRANSFER and chain[1] is TRANSFER (not CASH_OUT)
        # But chain[1] is TRANSFER and chain[2] is CASH_OUT -> transfer_then_cashout
        assert "transfer_then_cashout" in patterns

    def test_sudden_type_change(self):
        """Non-high-risk -> high-risk type triggers sudden_type_change."""
        chain = [
            make_txn(txn_type="PAYMENT"),
            make_txn(txn_type="CASH_OUT"),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        patterns = [a["pattern"] for a in anomalies]
        assert "sudden_type_change" in patterns

    def test_no_type_change_within_high_risk(self):
        """TRANSFER -> CASH_OUT should NOT produce sudden_type_change
        because TRANSFER is already a high-risk type."""
        chain = [
            make_txn(txn_type="TRANSFER"),
            make_txn(txn_type="CASH_OUT"),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        patterns = [a["pattern"] for a in anomalies]
        assert "sudden_type_change" not in patterns

    def test_non_high_risk_pair_no_anomaly(self):
        """PAYMENT -> DEBIT produces no anomalies."""
        chain = [
            make_txn(txn_type="PAYMENT"),
            make_txn(txn_type="DEBIT"),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        assert anomalies == []

    def test_cashout_then_transfer_no_transfer_then_cashout(self):
        """CASH_OUT -> TRANSFER does NOT trigger transfer_then_cashout."""
        chain = [
            make_txn(txn_type="CASH_OUT"),
            make_txn(txn_type="TRANSFER"),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        patterns = [a["pattern"] for a in anomalies]
        assert "transfer_then_cashout" not in patterns

    def test_total_amount_in_transfer_then_cashout(self):
        """Total amount is sum of the two transactions."""
        chain = [
            make_txn(txn_type="TRANSFER", amount=1000.0),
            make_txn(txn_type="CASH_OUT", amount=2000.0),
        ]
        anomalies = da.detect_transfer_then_cashout(chain)
        tc = [a for a in anomalies if a["pattern"] == "transfer_then_cashout"]
        assert tc[0]["total_amount"] == 3000.0


# ===================================================================
# Tests for detect_sudden_amount_increase
# ===================================================================

class TestDetectSuddenAmountIncrease:
    """Tests for detect_sudden_amount_increase(chain)."""

    def test_no_spike(self):
        """No spike when amounts are similar."""
        chain = [
            make_txn(amount=100.0),
            make_txn(amount=200.0),  # 2x, below 3x threshold
        ]
        assert da.detect_sudden_amount_increase(chain) == []

    def test_exactly_3x_triggers(self):
        """Exactly 3x should trigger (>= 3x)."""
        chain = [
            make_txn(amount=100.0),
            make_txn(amount=300.0),  # exactly 3x
        ]
        anomalies = da.detect_sudden_amount_increase(chain)
        assert len(anomalies) == 1
        assert anomalies[0]["pattern"] == "sudden_amount_increase"
        assert anomalies[0]["spike_ratio"] == 3.0

    def test_above_3x_triggers(self):
        """Above 3x triggers."""
        chain = [
            make_txn(amount=100.0),
            make_txn(amount=500.0),  # 5x
        ]
        anomalies = da.detect_sudden_amount_increase(chain)
        assert len(anomalies) == 1
        assert anomalies[0]["spike_ratio"] == 5.0

    def test_prev_amount_zero_no_trigger(self):
        """prev_amount == 0 should NOT trigger sudden_amount_increase."""
        chain = [
            make_txn(amount=0.0),
            make_txn(amount=50000.0),
        ]
        assert da.detect_sudden_amount_increase(chain) == []

    def test_multiple_spikes(self):
        """Multiple spikes in one chain."""
        chain = [
            make_txn(amount=100.0),
            make_txn(amount=400.0),  # 4x spike
            make_txn(amount=50.0),   # drop, no spike
            make_txn(amount=200.0),  # 4x spike
        ]
        anomalies = da.detect_sudden_amount_increase(chain)
        assert len(anomalies) == 2

    def test_spike_ratio_rounding(self):
        """Spike ratio is rounded to 2 decimal places."""
        chain = [
            make_txn(amount=100.0),
            make_txn(amount=333.0),  # 3.33x
        ]
        anomalies = da.detect_sudden_amount_increase(chain)
        assert len(anomalies) == 1
        assert anomalies[0]["spike_ratio"] == 3.33


# ===================================================================
# Tests for detect_destination_anomalies
# ===================================================================

class TestDetectDestinationAnomalies:
    """Tests for detect_destination_anomalies(dest_map)."""

    def test_empty_dest_map(self):
        assert da.detect_destination_anomalies({}) == []

    def test_two_transfers_no_trigger(self):
        """Exactly 2 transfers to same dest should NOT trigger (need >= 3)."""
        dest_map = {
            "D1": [
                make_txn(txn_type="TRANSFER", name_dest="D1"),
                make_txn(txn_type="TRANSFER", name_dest="D1"),
            ]
        }
        assert da.detect_destination_anomalies(dest_map) == []

    def test_three_transfers_triggers(self):
        """Exactly 3 transfers to same dest SHOULD trigger."""
        dest_map = {
            "D1": [
                make_txn(txn_type="TRANSFER", amount=1000.0, name_dest="D1", name_orig="C1"),
                make_txn(txn_type="TRANSFER", amount=2000.0, name_dest="D1", name_orig="C2"),
                make_txn(txn_type="TRANSFER", amount=3000.0, name_dest="D1", name_orig="C3"),
            ]
        }
        anomalies = da.detect_destination_anomalies(dest_map)
        assert len(anomalies) == 1
        assert anomalies[0]["pattern"] == "multiple_transfers_to_same_dest"
        assert anomalies[0]["transfer_count"] == 3
        assert anomalies[0]["total_inflow"] == 6000.0
        assert anomalies[0]["destination_account"] == "D1"
        assert anomalies[0]["senders"] == ["C1", "C2", "C3"]

    def test_non_transfer_types_excluded(self):
        """Only TRANSFER type counts; other types ignored."""
        dest_map = {
            "D1": [
                make_txn(txn_type="TRANSFER", name_dest="D1"),
                make_txn(txn_type="TRANSFER", name_dest="D1"),
                make_txn(txn_type="CASH_OUT", name_dest="D1"),  # not TRANSFER
            ]
        }
        assert da.detect_destination_anomalies(dest_map) == []

    def test_sorted_by_total_inflow_descending(self):
        """Results are sorted by total_inflow descending."""
        dest_map = {
            "D_low": [
                make_txn(txn_type="TRANSFER", amount=100.0, name_dest="D_low"),
                make_txn(txn_type="TRANSFER", amount=100.0, name_dest="D_low"),
                make_txn(txn_type="TRANSFER", amount=100.0, name_dest="D_low"),
            ],
            "D_high": [
                make_txn(txn_type="TRANSFER", amount=5000.0, name_dest="D_high"),
                make_txn(txn_type="TRANSFER", amount=5000.0, name_dest="D_high"),
                make_txn(txn_type="TRANSFER", amount=5000.0, name_dest="D_high"),
            ],
        }
        anomalies = da.detect_destination_anomalies(dest_map)
        assert len(anomalies) == 2
        assert anomalies[0]["destination_account"] == "D_high"
        assert anomalies[1]["destination_account"] == "D_low"


# ===================================================================
# Tests for calculate_risk_score
# ===================================================================

class TestCalculateRiskScore:
    """Tests for calculate_risk_score(anomalies, chain)."""

    def test_no_anomalies_no_risk_type(self):
        """No anomalies and non-high-risk types -> score 0."""
        chain = [make_txn(txn_type="PAYMENT"), make_txn(txn_type="DEBIT")]
        score = da.calculate_risk_score([], chain)
        assert score == 0

    def test_repeated_high_value_scoring(self):
        """repeated_high_value adds 25 points."""
        chain = [make_txn(txn_type="PAYMENT")]
        anomalies = [{"pattern": "repeated_high_value"}]
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 25

    def test_transfer_then_cashout_scoring(self):
        """transfer_then_cashout adds 30 points."""
        chain = [make_txn(txn_type="PAYMENT")]
        anomalies = [{"pattern": "transfer_then_cashout"}]
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 30

    def test_transfer_transfer_cashout_scoring(self):
        """transfer_transfer_cashout adds 35 points."""
        chain = [make_txn(txn_type="PAYMENT")]
        anomalies = [{"pattern": "transfer_transfer_cashout"}]
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 35

    def test_sudden_amount_increase_scoring(self):
        """sudden_amount_increase adds 15 points."""
        chain = [make_txn(txn_type="PAYMENT")]
        anomalies = [{"pattern": "sudden_amount_increase"}]
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 15

    def test_sudden_type_change_scoring(self):
        """sudden_type_change adds 10 points."""
        chain = [make_txn(txn_type="PAYMENT")]
        anomalies = [{"pattern": "sudden_type_change"}]
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 10

    def test_high_risk_type_adds_5_per_txn(self):
        """Each CASH_OUT or TRANSFER in chain adds +5."""
        chain = [
            make_txn(txn_type="TRANSFER"),
            make_txn(txn_type="CASH_OUT"),
            make_txn(txn_type="PAYMENT"),
        ]
        score = da.calculate_risk_score([], chain)
        assert score == 10  # 5 + 5 + 0

    def test_is_fraud_adds_20_per_txn(self):
        """Each isFraud==1 transaction adds +20."""
        chain = [
            make_txn(txn_type="PAYMENT", is_fraud=1),
            make_txn(txn_type="PAYMENT", is_fraud=0),
        ]
        score = da.calculate_risk_score([], chain)
        assert score == 20

    def test_combined_scoring(self):
        """Anomalies + chain types + fraud combine correctly."""
        chain = [
            make_txn(txn_type="TRANSFER", is_fraud=1),  # +5 (type) + 20 (fraud)
            make_txn(txn_type="CASH_OUT", is_fraud=0),   # +5 (type)
        ]
        anomalies = [
            {"pattern": "transfer_then_cashout"},  # +30
        ]
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 60  # 30 + 5 + 20 + 5

    def test_score_capped_at_100(self):
        """Score cannot exceed 100."""
        chain = [
            make_txn(txn_type="TRANSFER", is_fraud=1),
            make_txn(txn_type="CASH_OUT", is_fraud=1),
        ]
        anomalies = [
            {"pattern": "repeated_high_value"},        # +25
            {"pattern": "transfer_then_cashout"},      # +30
            {"pattern": "transfer_transfer_cashout"},  # +35
        ]
        # 25+30+35 + (5+20) + (5+20) = 140 -> capped to 100
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 100

    def test_chain_with_only_non_high_risk_types(self):
        """Chain with PAYMENT/DEBIT types adds no type-based points."""
        chain = [
            make_txn(txn_type="PAYMENT"),
            make_txn(txn_type="DEBIT"),
        ]
        anomalies = [{"pattern": "sudden_amount_increase"}]  # +15
        score = da.calculate_risk_score(anomalies, chain)
        assert score == 15


# ===================================================================
# Tests for classify_risk
# ===================================================================

class TestClassifyRisk:
    """Tests for classify_risk(score)."""

    def test_low_at_zero(self):
        assert da.classify_risk(0) == "LOW"

    def test_low_at_39(self):
        """Score 39 -> LOW (boundary: < 40)."""
        assert da.classify_risk(39) == "LOW"

    def test_medium_at_40(self):
        """Score 40 -> MEDIUM (boundary: >= 40)."""
        assert da.classify_risk(40) == "MEDIUM"

    def test_medium_at_55(self):
        assert da.classify_risk(55) == "MEDIUM"

    def test_medium_at_70(self):
        """Score 70 -> MEDIUM (boundary: <= 70)."""
        assert da.classify_risk(70) == "MEDIUM"

    def test_high_at_71(self):
        """Score 71 -> HIGH (boundary: > 70)."""
        assert da.classify_risk(71) == "HIGH"

    def test_high_at_100(self):
        assert da.classify_risk(100) == "HIGH"


# ===================================================================
# Tests for analyze (end-to-end)
# ===================================================================

class TestAnalyze:
    """Tests for analyze(transactions)."""

    def test_empty_transactions(self):
        """Empty transactions return empty results."""
        flagged, dest_anom, single = da.analyze([])
        assert flagged == []
        assert dest_anom == []
        assert single == []

    def test_single_transaction_no_chains(self):
        """A single transaction can't form a chain; gets checked as individual."""
        txns = [
            make_txn(
                row=2, txn_type="TRANSFER", amount=50000.0,
                old_balance_org=0.0, new_balance_orig=0.0,
                is_fraud=1,
            ),
        ]
        flagged, dest_anom, single = da.analyze(txns)
        assert flagged == []
        assert len(single) >= 1
        # Should have flags for high-value, high-risk type, and fraud
        flags_text = " ".join(single[0]["flags"])
        assert "High-value" in flags_text
        assert "fraud" in flags_text.lower()

    def test_chain_with_anomalies_flagged(self):
        """A chain with transfer->cashout anomaly appears in flagged_chains."""
        txns = [
            make_txn(
                row=2, txn_type="TRANSFER", amount=15000.0,
                old_balance_org=50000.0, new_balance_orig=35000.0,
            ),
            make_txn(
                row=3, txn_type="CASH_OUT", amount=20000.0,
                old_balance_org=35000.0, new_balance_orig=15000.0,
            ),
        ]
        flagged, dest_anom, single = da.analyze(txns)
        assert len(flagged) >= 1
        chain = flagged[0]
        patterns = [a["pattern"] for a in chain["anomalies"]]
        assert "transfer_then_cashout" in patterns
        assert "repeated_high_value" in patterns
        assert chain["risk_score"] > 0

    def test_flagged_chains_sorted_by_risk_desc(self):
        """Flagged chains are sorted by risk_score descending."""
        # Chain A: TRANSFER -> CASH_OUT (score ~ 30 + type points)
        # Chain B: has high-value repeated + TRANSFER->TRANSFER->CASH_OUT (higher score)
        txns = [
            # Chain A (lower risk)
            make_txn(
                row=2, txn_type="TRANSFER", amount=500.0,
                old_balance_org=10000.0, new_balance_orig=9500.0,
            ),
            make_txn(
                row=3, txn_type="CASH_OUT", amount=500.0,
                old_balance_org=9500.0, new_balance_orig=9000.0,
            ),
            # Chain B (higher risk)
            make_txn(
                row=4, txn_type="TRANSFER", amount=15000.0,
                old_balance_org=80000.0, new_balance_orig=65000.0,
                is_fraud=1,
            ),
            make_txn(
                row=5, txn_type="TRANSFER", amount=20000.0,
                old_balance_org=65000.0, new_balance_orig=45000.0,
            ),
            make_txn(
                row=6, txn_type="CASH_OUT", amount=30000.0,
                old_balance_org=45000.0, new_balance_orig=15000.0,
            ),
        ]
        flagged, _, _ = da.analyze(txns)
        assert len(flagged) == 2
        assert flagged[0]["risk_score"] >= flagged[1]["risk_score"]

    def test_destination_anomalies_in_analyze(self):
        """Destinations with >= 3 transfers appear in dest_anomalies."""
        txns = [
            make_txn(row=2, txn_type="TRANSFER", name_dest="SINK",
                     name_orig="C1", old_balance_org=0.0, new_balance_orig=0.0),
            make_txn(row=3, txn_type="TRANSFER", name_dest="SINK",
                     name_orig="C2", old_balance_org=0.0, new_balance_orig=0.0),
            make_txn(row=4, txn_type="TRANSFER", name_dest="SINK",
                     name_orig="C3", old_balance_org=0.0, new_balance_orig=0.0),
        ]
        _, dest_anom, _ = da.analyze(txns)
        assert len(dest_anom) == 1
        assert dest_anom[0]["destination_account"] == "SINK"

    def test_unchained_transactions_checked_individually(self):
        """Transactions not part of any chain are checked as singles."""
        txns = [
            make_txn(
                row=2, txn_type="CASH_OUT", amount=50000.0,
                old_balance_org=0.0, new_balance_orig=0.0,
                is_fraud=0,
            ),
        ]
        _, _, single = da.analyze(txns)
        assert len(single) == 1
        assert single[0]["transaction"]["row"] == 2

    def test_chained_transactions_excluded_from_singles(self):
        """Transactions that are part of a chain should NOT appear in single_txn_anomalies."""
        txns = [
            make_txn(
                row=2, txn_type="TRANSFER", amount=50000.0,
                old_balance_org=100000.0, new_balance_orig=50000.0,
                is_fraud=1,
            ),
            make_txn(
                row=3, txn_type="CASH_OUT", amount=50000.0,
                old_balance_org=50000.0, new_balance_orig=0.0,
                is_fraud=1,
            ),
        ]
        flagged, _, single = da.analyze(txns)
        assert len(flagged) >= 1
        single_rows = [s["transaction"]["row"] for s in single]
        assert 2 not in single_rows
        assert 3 not in single_rows

    def test_single_txn_amount_exceeds_balance_flag(self):
        """Single txn where amount > oldbalanceOrg gets 'exceeds balance' flag."""
        txns = [
            make_txn(
                row=2, txn_type="PAYMENT", amount=5000.0,
                old_balance_org=3000.0, new_balance_orig=0.0,
            ),
        ]
        _, _, single = da.analyze(txns)
        assert len(single) == 1
        flags_text = " ".join(single[0]["flags"])
        assert "exceeds" in flags_text.lower()

    def test_single_txn_zero_old_balance_no_exceeds_flag(self):
        """Single txn with oldbalanceOrg==0 does NOT get 'exceeds balance' flag."""
        txns = [
            make_txn(
                row=2, txn_type="TRANSFER", amount=5000.0,
                old_balance_org=0.0, new_balance_orig=0.0,
            ),
        ]
        _, _, single = da.analyze(txns)
        assert len(single) == 1
        flags_text = " ".join(single[0]["flags"])
        assert "exceeds" not in flags_text.lower()

    def test_analyze_returns_three_element_tuple(self):
        """analyze() returns a 3-tuple."""
        result = da.analyze([])
        assert isinstance(result, tuple)
        assert len(result) == 3
