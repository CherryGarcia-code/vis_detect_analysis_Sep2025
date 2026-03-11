"""Quick smoke test for the GLM-HMM implementation."""
import sys
sys.path.insert(0, 'src')

import numpy as np
from scipy.special import expit
from visdetect.analysis.hmm import GLMHMM, GLMHMMConfig

np.random.seed(42)
T = 200

# Ground truth: 2 states
true_weights = np.array([[-2.0, 1.0, 0.0, 0.0],
                          [ 1.5, 0.2, 0.3, 0.0]])
true_A = np.array([[0.95, 0.05],
                    [0.08, 0.92]])
true_pi = np.array([0.5, 0.5])

# Generate latent states
z = np.zeros(T, dtype=int)
z[0] = np.random.choice(2, p=true_pi)
for t in range(1, T):
    z[t] = np.random.choice(2, p=true_A[z[t-1]])

# Generate observations
X = np.column_stack([
    np.ones(T),
    np.random.uniform(0, 2, T),
    np.random.binomial(1, 0.5, T).astype(float),
    np.random.binomial(1, 0.3, T).astype(float),
])
y = np.array([np.random.binomial(1, expit(true_weights[z[t]] @ X[t]))
              for t in range(T)], dtype=float)

sessions_data = [{"y": y, "X": X, "df": None,
                  "session_name": "test", "feature_names": ["b","s","c","r"]}]

# Fit
cfg = GLMHMMConfig(max_iter=100, n_restarts=5, verbose=False)
model = GLMHMM(n_states=2, n_features=4, config=cfg)
ll = model.fit(sessions_data, seed=0)
model.sort_states_by_bias()

print(f"Final LL: {ll:.2f}")
print(f"Recovered weights:\n{model.weights}")
print(f"True weights:\n{true_weights}")
print(f"Recovered A:\n{np.round(model.transition_matrix, 3)}")
print(f"True A:\n{true_A}")

states = model.most_likely_states(sessions_data[0])
accuracy = np.mean(states == z)
accuracy = max(accuracy, 1 - accuracy)
print(f"State recovery accuracy: {accuracy:.2f}")
print(f"BIC: {model.bic(sessions_data):.2f}")
print(f"\nSMOKE TEST PASSED" if accuracy > 0.6 else "\nSMOKE TEST FAILED")
