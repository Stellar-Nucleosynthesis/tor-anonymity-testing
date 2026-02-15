from typing import Dict, List

def compute_guard_exit_deanon_probability(guard_fraction: float,
                                        exit_fraction: float,
                                        guard_exit_correlation: float = 1.0) -> float:
    """
    Compute theoretical probability of deanonymization.

    For Guard+Exit attack, the probability that a circuit is compromised
    is approximately: P(guard) * P(exit) * P(correlation)

    Args:
        guard_fraction: Fraction of guards controlled
        exit_fraction: Fraction of exits controlled
        guard_exit_correlation: Probability of successful correlation

    Returns:
        Probability of deanonymization
    """
    return guard_fraction * exit_fraction * guard_exit_correlation


def compute_circuit_compromise_rate(circuits: List[Dict],
                                    adversary_guards: List[str],
                                    adversary_exits: List[str]) -> Dict[str, float]:
    """
    Compute what fraction of circuits are compromised.

    Args:
        circuits: List of circuit data
        adversary_guards: List of adversary guard IDs
        adversary_exits: List of adversary exit IDs

    Returns:
        Dictionary with compromise statistics
    """
    total_circuits = len(circuits)
    guard_compromised = 0
    exit_compromised = 0
    both_compromised = 0

    for circuit in circuits:
        guard_comp = circuit.get('guard') in adversary_guards
        exit_comp = circuit.get('exit') in adversary_exits

        if guard_comp:
            guard_compromised += 1
        if exit_comp:
            exit_compromised += 1
        if guard_comp and exit_comp:
            both_compromised += 1

    return {
        'total_circuits': total_circuits,
        'guard_compromise_rate': guard_compromised / total_circuits,
        'exit_compromise_rate': exit_compromised / total_circuits,
        'full_compromise_rate': both_compromised / total_circuits,
        'guard_compromised_count': guard_compromised,
        'exit_compromised_count': exit_compromised,
        'both_compromised_count': both_compromised
    }