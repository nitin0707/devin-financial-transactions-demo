"""
Transaction Risk Scoring and Classification Module

Analyzes financial transactions from Example1.csv and assigns fraud risk scores
based on the following risk classification guidelines:

1. Transactions above 10,000 are high risk.
2. CASH_OUT and TRANSFER are higher risk transaction types.
3. Transactions to new or previously unseen destination accounts are risky.
4. Rapid sequence of transactions from the same account increases risk.
5. Fraudulent transactions often involve high amounts followed by cash-out.

Risk Categories:
  - LOW:    score < 40
  - MEDIUM: score between 40 and 70
  - HIGH:   score > 70
"""

import csv
import os
from collections import defaultdict


def load_transactions(filepath):
    """Load transactions from a CSV file and return a list of dicts."""
    transactions = []
    with open(filepath, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["step"] = int(row["step"])
            row["amount"] = float(row["amount"])
            row["oldbalanceOrg"] = float(row["oldbalanceOrg"])
            row["newbalanceOrig"] = float(row["newbalanceOrig"])
            row["oldbalanceDest"] = float(row["oldbalanceDest"])
            row["newbalanceDest"] = float(
                row["newbalanceDest"].replace("E", "e")
            )
            row["isFraud"] = int(row["isFraud"])
            row["isFlaggedFraud"] = int(row["isFlaggedFraud"])
            transactions.append(row)
    return transactions


def compute_risk_scores(transactions):
    """
    Compute a risk score (0-100) for each transaction based on multiple
    risk factors, and assign a risk category (LOW, MEDIUM, HIGH).

    Risk factors and their max contributions:
      - High amount (>10,000):              up to 25 points
      - Risky transaction type:             up to 20 points
      - New/unseen destination account:     up to 15 points
      - Rapid transaction sequence:         up to 20 points
      - High amount followed by cash-out:   up to 20 points

    Total max = 100 points.
    """
    HIGH_RISK_TYPES = {"CASH_OUT", "TRANSFER"}
    AMOUNT_THRESHOLD = 10000

    # Pre-compute destination account first-seen index
    dest_first_seen = {}
    for idx, txn in enumerate(transactions):
        dest = txn["nameDest"]
        if dest not in dest_first_seen:
            dest_first_seen[dest] = idx

    # Pre-compute per-origin account transaction indices for rapid sequence detection
    origin_txn_indices = defaultdict(list)
    for idx, txn in enumerate(transactions):
        origin_txn_indices[txn["nameOrig"]].append(idx)

    # Build a lookup: for each origin account, track the sequence of transaction
    # types and amounts to detect high-amount-then-cashout patterns
    origin_txn_sequence = defaultdict(list)
    for idx, txn in enumerate(transactions):
        origin_txn_sequence[txn["nameOrig"]].append(
            {"index": idx, "type": txn["type"], "amount": txn["amount"]}
        )

    scored_transactions = []

    for idx, txn in enumerate(transactions):
        score = 0
        risk_factors = []

        # --- Factor 1: High amount (up to 25 points) ---
        if txn["amount"] > AMOUNT_THRESHOLD:
            # Scale: 10k->15pts, 100k->20pts, 500k+->25pts
            if txn["amount"] > 500000:
                amount_score = 25
            elif txn["amount"] > 100000:
                amount_score = 20
            elif txn["amount"] > 50000:
                amount_score = 18
            else:
                amount_score = 15
            score += amount_score
            risk_factors.append(
                f"High amount: {txn['amount']:.2f} (+{amount_score})"
            )

        # --- Factor 2: Risky transaction type (up to 20 points) ---
        if txn["type"] in HIGH_RISK_TYPES:
            if txn["type"] == "CASH_OUT":
                type_score = 20
            else:
                type_score = 15
            score += type_score
            risk_factors.append(
                f"Risky type: {txn['type']} (+{type_score})"
            )

        # --- Factor 3: New/unseen destination account (up to 15 points) ---
        if dest_first_seen[txn["nameDest"]] == idx:
            dest_score = 15
            score += dest_score
            risk_factors.append(
                f"New destination: {txn['nameDest']} (+{dest_score})"
            )

        # --- Factor 4: Rapid sequence from same origin (up to 20 points) ---
        origin = txn["nameOrig"]
        txn_indices = origin_txn_indices[origin]
        if len(txn_indices) > 1:
            # Check how many transactions from this origin share the same step
            same_step_count = sum(
                1
                for i in txn_indices
                if transactions[i]["step"] == txn["step"]
            )
            if same_step_count >= 4:
                rapid_score = 20
            elif same_step_count >= 3:
                rapid_score = 15
            elif same_step_count >= 2:
                rapid_score = 10
            else:
                rapid_score = 0

            if rapid_score > 0:
                score += rapid_score
                risk_factors.append(
                    f"Rapid sequence: {same_step_count} txns from {origin} "
                    f"in step {txn['step']} (+{rapid_score})"
                )

        # --- Factor 5: High amount followed by cash-out pattern (up to 20 points) ---
        seq = origin_txn_sequence[origin]
        position_in_seq = next(
            i for i, s in enumerate(seq) if s["index"] == idx
        )
        if txn["type"] == "CASH_OUT" and position_in_seq > 0:
            prev = seq[position_in_seq - 1]
            if prev["amount"] > AMOUNT_THRESHOLD:
                cashout_score = 20
                score += cashout_score
                risk_factors.append(
                    f"Cash-out after high amount ({prev['amount']:.2f}) "
                    f"(+{cashout_score})"
                )

        # Cap score at 100
        score = min(score, 100)

        # Assign category
        if score > 70:
            category = "HIGH"
        elif score >= 40:
            category = "MEDIUM"
        else:
            category = "LOW"

        scored_transactions.append(
            {
                **txn,
                "risk_score": score,
                "risk_category": category,
                "risk_factors": "; ".join(risk_factors) if risk_factors else "None",
            }
        )

    return scored_transactions


def generate_risk_report(scored_transactions, output_path):
    """Generate a transaction-level risk report as a CSV file."""
    fieldnames = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
        "risk_score",
        "risk_category",
        "risk_factors",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_transactions)

    return output_path


def print_summary(scored_transactions):
    """Print a summary of the risk analysis to stdout."""
    total = len(scored_transactions)
    low = sum(1 for t in scored_transactions if t["risk_category"] == "LOW")
    medium = sum(1 for t in scored_transactions if t["risk_category"] == "MEDIUM")
    high = sum(1 for t in scored_transactions if t["risk_category"] == "HIGH")

    actual_fraud = sum(1 for t in scored_transactions if t["isFraud"] == 1)
    high_and_fraud = sum(
        1
        for t in scored_transactions
        if t["risk_category"] == "HIGH" and t["isFraud"] == 1
    )

    print("=" * 70)
    print("TRANSACTION RISK ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total transactions analyzed: {total}")
    print(f"  LOW  risk (score < 40):      {low:>4} ({low/total*100:.1f}%)")
    print(f"  MEDIUM risk (40 <= score <= 70): {medium:>4} ({medium/total*100:.1f}%)")
    print(f"  HIGH risk (score > 70):      {high:>4} ({high/total*100:.1f}%)")
    print()
    print(f"Actual fraudulent transactions (isFraud=1): {actual_fraud}")
    print(f"Fraudulent transactions flagged as HIGH risk: {high_and_fraud}")
    print()

    # Show top 10 highest risk transactions
    top_risk = sorted(
        scored_transactions, key=lambda x: x["risk_score"], reverse=True
    )[:10]
    print("TOP 10 HIGHEST RISK TRANSACTIONS:")
    print("-" * 70)
    print(
        f"{'Type':<10} {'Amount':>14} {'Origin':<14} {'Destination':<14} "
        f"{'Score':>5} {'Category':<8} {'Fraud':>5}"
    )
    print("-" * 70)
    for t in top_risk:
        print(
            f"{t['type']:<10} {t['amount']:>14,.2f} {t['nameOrig']:<14} "
            f"{t['nameDest']:<14} {t['risk_score']:>5} {t['risk_category']:<8} "
            f"{'YES' if t['isFraud'] == 1 else 'NO':>5}"
        )
    print("=" * 70)


def main():
    """Main entry point: load data, score transactions, generate report."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "data", "Example1.csv")
    output_path = os.path.join(base_dir, "reports", "transaction_risk_report.csv")

    # Ensure reports directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading transactions from: {input_path}")
    transactions = load_transactions(input_path)
    print(f"Loaded {len(transactions)} transactions.")
    print()

    print("Computing risk scores...")
    scored = compute_risk_scores(transactions)

    print(f"Generating risk report: {output_path}")
    generate_risk_report(scored, output_path)
    print()

    print_summary(scored)

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
