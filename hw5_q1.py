from matplotlib import pyplot as plt
import numpy as np

# Generate data
N = 500
x = np.random.rand(N)

pi0 = np.array([0.7, 0.3])
w0 = np.array([-2, 1])
b0 = np.array([0.5, -0.5])
sigma0 = np.array([0.4, 0.3])

y = np.zeros_like(x)
for i in range(N):
    k = 0 if np.random.rand() < pi0[0] else 1
    y[i] = w0[k] * x[i] + b0[k] + np.random.randn() * sigma0[k]

# EM parameters
K = 2
pi = np.array([0.5, 0.5])
w = np.array([1.0, -1.0])
b = np.array([0.0, 0.0])
sigma = np.array([np.std(y), np.std(y)])

X = np.vstack([x, np.ones(N)]).T
log_likelihoods = []
tolerance = 1e-4
max_iterations = 100

# Run EM Algorithm 
for _ in range(max_iterations):
    # E-Step
    gamma = np.array([
        pi[k] * np.exp(-0.5 * ((y - (w[k] * x + b[k]))**2) / sigma[k]**2) / (np.sqrt(2 * np.pi) * sigma[k])
        for k in range(K)
    ]).T
    gamma /= gamma.sum(axis=1, keepdims=True)

    # M-Step
    samples_per_component = gamma.sum(axis=0)
    pi = samples_per_component / N

    for k in range(K):
        W = np.diag(gamma[:, k])
        theta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        w[k], b[k] = theta
        y_hat = X @ theta
        sigma[k] = np.sqrt(((gamma[:, k] * (y - y_hat)**2).sum()) / samples_per_component[k])

    # Log-likelihood
    log_likelihood = np.sum(np.log(np.sum([
        pi[k] * np.exp(-0.5 * ((y - (w[k] * x + b[k]))**2) / sigma[k]**2) / (np.sqrt(2 * np.pi) * sigma[k])
        for k in range(K)
    ], axis=0)))
    log_likelihoods.append(log_likelihood)

    if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance: break

# Output results
print(f"\nConverged after {len(log_likelihoods)} iterations")
for k in range(K):
    print(f"Component {k+1}: pi = {pi[k]:.4f}, w = {w[k]:.4f}, b = {b[k]:.4f}, sigma = {sigma[k]:.4f}")
print()

# Plot data and fitted lines
x_grid = np.linspace(0, 1, 100)

plt.figure()
plt.scatter(x, y, color='red', s=10)
for k in range(K):
    plt.plot(x_grid, w[k] * x_grid + b[k], label=f"Component {k+1}")
plt.title("Data and Fitted Lines")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot marginal log-likelihood over iterations
plt.figure()
plt.plot(log_likelihoods, marker='o')
plt.title("Log-Likelihood Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.show()
