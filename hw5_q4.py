import itertools as it
import numpy as np

# State space
states = [0, 1, 2]

# Observation sequence 0101
observations = [0, 1, 0, 1]

# HMM paramaters
A = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])
B = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
pi = np.array([0.5, 0.3, 0.2])

results = []

# Run HMM
for state_sequence in it.product(states, repeat=len(observations)):
    prior = pi[state_sequence[0]]
    for i in range(1, len(observations)):
        prior *= A[state_sequence[i - 1], state_sequence[i]]

    likelihood = 1.0
    for i in range(len(observations)):
        likelihood *= B[state_sequence[i], observations[i]]

    posterior = prior * likelihood

    results.append((state_sequence, prior, likelihood, posterior))

# Sort by posterior and take top 3
results.sort(key=lambda x: x[3], reverse=True)
top_three = results[:3]
total_posterior = sum(x[3] for x in results)
top_three_normalized = [(x[0], x[1], x[2], x[3] / total_posterior) for x in top_three]

# Output results
print("\nMost Probable Hidden State Sequences  | Prior   | Likelihood  | Posterior")
for sequence, prior_prob, likelihood, posterior_prob in top_three_normalized:
    print(f"{sequence}                          | {prior_prob:.4f}  | {likelihood:.4f}      | {posterior_prob:.4f}")
print()
