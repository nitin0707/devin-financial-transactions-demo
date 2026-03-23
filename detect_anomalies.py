"""
Anomalous Transaction Sequence Detection

Analyzes transaction data to identify suspicious patterns:
1. Repeated high-value transactions in linked account sequences
2. Transfer followed by cash-out sequences
3. Sudden increases in transaction amounts

Transactions are linked into sequences by matching balance continuity
(newbalanceOrig of one transaction == oldbalanceOrg of the next),
which reveals the true transaction chains even when customer IDs differ.

Risk Scoring:
- LOW: score < 40
- MEDIUM: score between 40 and 70
- HIGH: score > 70
"""

import csv
import json
import os
from collections import defaultdict

HIGH_VALUE_THRESHOLD = 10000
AMOUNT_SPIKE_MULTIPLIER = 3
HIGH_RISK_TYPES = {"CASH_OUT", "TRANSFER"}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "Example1.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "anomaly_report.json")


def load_transactions(filepath):
    """Load all transactions from CSV."""
    transactions = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if not row.get("nameOrig"):
                continue
            transactions.append(
                {
                    "row": idx + 2,
                    "step": int(row["step"]),
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "nameOrig": row["nameOrig"],
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "nameDest": row["nameDest"],
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(
                        row["newbalanceDest"].replace("E", "e")
                    ),
                    "isFraud": int(row["isFraud"]),
                    "isFlaggedFraud": int(row["isFlaggedFraud"]),
                }
            )
    return transactions


def build_balance_chains(transactions):
    """
    Link transactions into sequences by balance continuity.
    When newbalanceOrig of transaction A matches oldbalanceOrg of
    transaction B (and both are > 0), they represent consecutive
    operations on the same underlying account.
    """
    chains = []
    used = set()
    by_old_balance = defaultdict(list)
    for i, txn in enumerate(transactions):
        if txn["oldbalanceOrg"] > 0:
            by_old_balance[txn["oldbalanceOrg"]].append(i)

    for start_idx, txn in enumerate(transactions):
        if start_idx in used:
            continue
        chain = [txn]
        used.add(start_idx)
        current = txn
        while True:
            new_bal = current["newbalanceOrig"]
            if new_bal <= 0:
                break
            found_next = False
            for next_idx in by_old_balance.get(new_bal, []):
                if next_idx not in used:
                    next_txn = transactions[next_idx]
                    chain.append(next_txn)
                    used.add(next_idx)
                    current = next_txn
                    found_next = True
                    break
            if not found_next:
                break
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def build_destination_map(transactions):
    """Map destination accounts to all transactions they receive."""
    dest_map = defaultdict(list)
    for txn in transactions:
        dest_map[txn["nameDest"]].append(txn)
    return dest_map


def _format_txns(txns):
    """Format a list of transactions for report output."""
    return [
        {
            "row": t["row"],
            "customer": t["nameOrig"],
            "type": t["type"],
            "amount": t["amount"],
            "destination": t["nameDest"],
            "isFraud": t["isFraud"],
        }
        for t in txns
    ]


def detect_repeated_high_value(chain):
    """Detect repeated high-value transactions in a linked sequence."""
    anomalies = []
    consecutive_high = []
    for txn in chain:
        if txn["amount"] > HIGH_VALUE_THRESHOLD:
            consecutive_high.append(txn)
        else:
            if len(consecutive_high) >= 2:
                anomalies.append(
                    {
                        "pattern": "repeated_high_value",
                        "description": (
                            f"{len(consecutive_high)} consecutive high-value "
                            f"transactions (>{HIGH_VALUE_THRESHOLD:,}) "
                            f"in sequence"
                        ),
                        "transactions": _format_txns(consecutive_high),
                        "total_amount": round(
                            sum(t["amount"] for t in consecutive_high), 2
                        ),
                    }
                )
            consecutive_high = []
    if len(consecutive_high) >= 2:
        anomalies.append(
            {
                "pattern": "repeated_high_value",
                "description": (
                    f"{len(consecutive_high)} consecutive high-value "
                    f"transactions (>{HIGH_VALUE_THRESHOLD:,}) in sequence"
                ),
                "transactions": _format_txns(consecutive_high),
                "total_amount": round(
                    sum(t["amount"] for t in consecutive_high), 2
                ),
            }
        )
    return anomalies


def detect_transfer_then_cashout(chain):
    """Detect TRANSFER->CASH_OUT, TRANSFER->TRANSFER->CASH_OUT,
    and sudden type changes to high-risk types."""
    anomalies = []
    for i in range(len(chain) - 1):
        if (
            chain[i]["type"] == "TRANSFER"
            and chain[i + 1]["type"] == "CASH_OUT"
        ):
            anomalies.append(
                {
                    "pattern": "transfer_then_cashout",
                    "description": "TRANSFER followed by CASH_OUT detected",
                    "transactions": _format_txns(
                        [chain[i], chain[i + 1]]
                    ),
                    "total_amount": round(
                        chain[i]["amount"] + chain[i + 1]["amount"], 2
                    ),
                }
            )
        if (
            i + 2 < len(chain)
            and chain[i]["type"] == "TRANSFER"
            and chain[i + 1]["type"] == "TRANSFER"
            and chain[i + 2]["type"] == "CASH_OUT"
        ):
            anomalies.append(
                {
                    "pattern": "transfer_transfer_cashout",
                    "description": (
                        "TRANSFER -> TRANSFER -> CASH_OUT "
                        "sequence detected"
                    ),
                    "transactions": _format_txns(chain[i : i + 3]),
                    "total_amount": round(
                        sum(
                            chain[j]["amount"] for j in range(i, i + 3)
                        ),
                        2,
                    ),
                }
            )
        if (
            chain[i]["type"] not in HIGH_RISK_TYPES
            and chain[i + 1]["type"] in HIGH_RISK_TYPES
        ):
            anomalies.append(
                {
                    "pattern": "sudden_type_change",
                    "description": (
                        f"Sudden change from {chain[i]['type']} to "
                        f"{chain[i + 1]['type']} (high-risk type)"
                    ),
                    "transactions": _format_txns(
                        [chain[i], chain[i + 1]]
                    ),
                    "total_amount": round(
                        chain[i]["amount"] + chain[i + 1]["amount"], 2
                    ),
                }
            )
    return anomalies


def detect_sudden_amount_increase(chain):
    """Detect sudden spikes where amount >= AMOUNT_SPIKE_MULTIPLIER
    times the previous transaction amount."""
    anomalies = []
    for i in range(1, len(chain)):
        prev_amount = chain[i - 1]["amount"]
        curr_amount = chain[i]["amount"]
        if (
            prev_amount > 0
            and curr_amount >= AMOUNT_SPIKE_MULTIPLIER * prev_amount
        ):
            anomalies.append(
                {
                    "pattern": "sudden_amount_increase",
                    "description": (
                        f"Amount spiked from {prev_amount:,.2f} to "
                        f"{curr_amount:,.2f} "
                        f"({curr_amount / prev_amount:.1f}x increase)"
                    ),
                    "transactions": _format_txns(
                        [chain[i - 1], chain[i]]
                    ),
                    "spike_ratio": round(curr_amount / prev_amount, 2),
                }
            )
    return anomalies


def detect_destination_anomalies(dest_map):
    """Detect destination accounts receiving suspiciously many transfers."""
    anomalies = []
    for dest, txns in dest_map.items():
        transfers_in = [t for t in txns if t["type"] == "TRANSFER"]
        if len(transfers_in) >= 3:
            total_inflow = sum(t["amount"] for t in transfers_in)
            anomalies.append(
                {
                    "destination_account": dest,
                    "pattern": "multiple_transfers_to_same_dest",
                    "description": (
                        f"Account {dest} received "
                        f"{len(transfers_in)} transfers "
                        f"totaling ${total_inflow:,.2f}"
                    ),
                    "transfer_count": len(transfers_in),
                    "total_inflow": round(total_inflow, 2),
                    "senders": [t["nameOrig"] for t in transfers_in],
                }
            )
    anomalies.sort(key=lambda x: x["total_inflow"], reverse=True)
    return anomalies


def calculate_risk_score(anomalies, chain):
    """
    Calculate risk score (0-100) based on anomalies.

    Scoring:
    - repeated_high_value: +25
    - transfer_then_cashout: +30
    - transfer_transfer_cashout: +35
    - sudden_amount_increase: +15
    - sudden_type_change: +10
    - Each CASH_OUT or TRANSFER in chain: +5
    - Any isFraud=1 transaction: +20
    """
    score = 0
    for anomaly in anomalies:
        p = anomaly["pattern"]
        if p == "repeated_high_value":
            score += 25
        elif p == "transfer_then_cashout":
            score += 30
        elif p == "transfer_transfer_cashout":
            score += 35
        elif p == "sudden_amount_increase":
            score += 15
        elif p == "sudden_type_change":
            score += 10
    for txn in chain:
        if txn["type"] in HIGH_RISK_TYPES:
            score += 5
        if txn["isFraud"] == 1:
            score += 20
    return min(score, 100)


def classify_risk(score):
    """Classify risk level based on score thresholds."""
    if score < 40:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"


def analyze(transactions):
    """Run all anomaly detection analyses and return structured results."""
    chains = build_balance_chains(transactions)
    dest_map = build_destination_map(transactions)

    chain_results = []
    for chain_idx, chain in enumerate(chains):
        anomalies = []
        anomalies.extend(detect_repeated_high_value(chain))
        anomalies.extend(detect_transfer_then_cashout(chain))
        anomalies.extend(detect_sudden_amount_increase(chain))
        risk_score = calculate_risk_score(anomalies, chain)
        risk_level = classify_risk(risk_score)
        chain_results.append(
            {
                "chain_id": f"chain_{chain_idx + 1}",
                "customers": [t["nameOrig"] for t in chain],
                "transaction_count": len(chain),
                "total_volume": round(
                    sum(t["amount"] for t in chain), 2
                ),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "anomalies": anomalies,
                "transactions": _format_txns(chain),
            }
        )

    flagged_chains = sorted(
        [c for c in chain_results if c["anomalies"]],
        key=lambda x: x["risk_score"],
        reverse=True,
    )
    dest_anomalies = detect_destination_anomalies(dest_map)

    chained_rows = set()
    for chain in chains:
        for txn in chain:
            chained_rows.add(txn["row"])

    single_txn_anomalies = []
    for txn in transactions:
        if txn["row"] in chained_rows:
            continue
        flags = []
        score = 0
        if txn["amount"] > HIGH_VALUE_THRESHOLD:
            flags.append(
                f"High-value transaction: ${txn['amount']:,.2f}"
            )
            score += 15
        if txn["type"] in HIGH_RISK_TYPES:
            flags.append(
                f"High-risk transaction type: {txn['type']}"
            )
            score += 5
        if txn["isFraud"] == 1:
            flags.append("Flagged as fraud in source data")
            score += 20
        if (
            txn["oldbalanceOrg"] > 0
            and txn["amount"] > txn["oldbalanceOrg"]
        ):
            flags.append(
                f"Amount ({txn['amount']:,.2f}) exceeds "
                f"balance ({txn['oldbalanceOrg']:,.2f})"
            )
            score += 10
        if flags:
            single_txn_anomalies.append(
                {
                    "customer": txn["nameOrig"],
                    "transaction": _format_txns([txn])[0],
                    "flags": flags,
                    "risk_score": min(score, 100),
                    "risk_level": classify_risk(min(score, 100)),
                }
            )
    single_txn_anomalies.sort(
        key=lambda x: x["risk_score"], reverse=True
    )
    return flagged_chains, dest_anomalies, single_txn_anomalies


def print_report(flagged_chains, dest_anomalies, single_txn_anomalies):
    """Print a human-readable summary."""
    print("=" * 80)
    print("ANOMALOUS TRANSACTION SEQUENCE DETECTION REPORT")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("SECTION 1: LINKED TRANSACTION CHAIN ANOMALIES")
    print("=" * 80)
    print(
        "\nTransactions linked by balance continuity "
        "(newbalanceOrig -> oldbalanceOrg)"
    )

    if not flagged_chains:
        print("\n  No chain anomalies detected.")
    else:
        high = [
            c for c in flagged_chains if c["risk_level"] == "HIGH"
        ]
        medium = [
            c for c in flagged_chains if c["risk_level"] == "MEDIUM"
        ]
        low = [
            c for c in flagged_chains if c["risk_level"] == "LOW"
        ]
        print(f"\n  Flagged chains: {len(flagged_chains)}")
        print(f"    HIGH risk:   {len(high)}")
        print(f"    MEDIUM risk: {len(medium)}")
        print(f"    LOW risk:    {len(low)}")

        for level_name, level_chains in [
            ("HIGH", high),
            ("MEDIUM", medium),
            ("LOW", low),
        ]:
            if not level_chains:
                continue
            print(f"\n  {'=' * 76}")
            print(f"  {level_name} RISK CHAINS")
            print(f"  {'=' * 76}")
            for chain in level_chains:
                print(f"\n  {chain['chain_id']}:")
                print(
                    f"    Risk Score: {chain['risk_score']} "
                    f"({chain['risk_level']})"
                )
                cust_str = ", ".join(chain["customers"])
                print(f"    Customers: {cust_str}")
                print(
                    f"    Transactions: "
                    f"{chain['transaction_count']}"
                )
                print(
                    f"    Total Volume: "
                    f"${chain['total_volume']:,.2f}"
                )
                print(
                    f"    Anomalies "
                    f"({len(chain['anomalies'])}):"
                )
                for anomaly in chain["anomalies"]:
                    print(
                        f"      - [{anomaly['pattern']}] "
                        f"{anomaly['description']}"
                    )
                    for txn in anomaly["transactions"]:
                        fraud = (
                            " **FRAUD**"
                            if txn["isFraud"]
                            else ""
                        )
                        print(
                            f"          Row {txn['row']:3d}: "
                            f"{txn['customer']:15s} "
                            f"{txn['type']:12s} "
                            f"${txn['amount']:>12,.2f} -> "
                            f"{txn['destination']}{fraud}"
                        )

    print("\n" + "=" * 80)
    print("SECTION 2: DESTINATION ACCOUNT ANOMALIES")
    print("=" * 80)
    print("\nAccounts receiving suspiciously many transfers")
    if not dest_anomalies:
        print("\n  No destination anomalies detected.")
    else:
        for anomaly in dest_anomalies:
            print(
                f"\n  Account: "
                f"{anomaly['destination_account']}"
            )
            print(
                f"    Transfers received: "
                f"{anomaly['transfer_count']}"
            )
            print(
                f"    Total inflow: "
                f"${anomaly['total_inflow']:,.2f}"
            )
            senders = ", ".join(anomaly["senders"])
            print(f"    Senders: {senders}")

    print("\n" + "=" * 80)
    print("SECTION 3: INDIVIDUAL HIGH-RISK TRANSACTIONS")
    print("=" * 80)
    print("\nIsolated transactions with suspicious characteristics")
    if not single_txn_anomalies:
        print("\n  No individual anomalies detected.")
    else:
        for item in single_txn_anomalies:
            txn = item["transaction"]
            fraud = " **FRAUD**" if txn["isFraud"] else ""
            print(
                f"\n  Customer: {item['customer']} "
                f"(Risk: {item['risk_score']} - "
                f"{item['risk_level']})"
            )
            print(
                f"    Row {txn['row']:3d}: "
                f"{txn['type']:12s} "
                f"${txn['amount']:>12,.2f} -> "
                f"{txn['destination']}{fraud}"
            )
            for flag in item["flags"]:
                print(f"      - {flag}")
    print(f"\n{'=' * 80}")


def main():
    print(f"Loading transactions from: {INPUT_FILE}")
    transactions = load_transactions(INPUT_FILE)
    print(f"Loaded {len(transactions)} transactions\n")
    flagged_chains, dest_anomalies, single_txn_anomalies = analyze(
        transactions
    )
    print_report(flagged_chains, dest_anomalies, single_txn_anomalies)

    report = {
        "summary": {
            "total_transactions": len(transactions),
            "flagged_chains": len(flagged_chains),
            "high_risk_chains": len(
                [
                    c
                    for c in flagged_chains
                    if c["risk_level"] == "HIGH"
                ]
            ),
            "medium_risk_chains": len(
                [
                    c
                    for c in flagged_chains
                    if c["risk_level"] == "MEDIUM"
                ]
            ),
            "low_risk_chains": len(
                [
                    c
                    for c in flagged_chains
                    if c["risk_level"] == "LOW"
                ]
            ),
            "destination_anomalies": len(dest_anomalies),
            "individual_anomalies": len(single_txn_anomalies),
        },
        "chain_anomalies": flagged_chains,
        "destination_anomalies": dest_anomalies,
        "individual_anomalies": single_txn_anomalies,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed JSON report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
