import numpy as np

rng = np.random.default_rng(42)

# --- Classification dataset for Logistic Regression ---
# 2 informative features, linearly separable-ish with noise
n = 300
X_cls = rng.normal(0, 1.5, size=(n, 2))
true_w = np.array([2.5, -1.8])
true_b = 0.3
logits = X_cls @ true_w + true_b + rng.normal(0, 1.0, size=n)
y_cls = (logits > 0).astype(int)

np.savetxt("classification_X.csv", X_cls, delimiter=",", fmt="%.8f")
np.savetxt("classification_y.csv", y_cls, delimiter=",", fmt="%d")

# --- Regression dataset for Ridge/Lasso ---
# 10 features, only 3 truly informative, rest are noise -> good for showing Lasso sparsity
n2 = 200
p = 10
X_reg = rng.normal(0, 1.0, size=(n2, p))
true_coef = np.zeros(p)
true_coef[[0, 3, 7]] = [4.0, -3.0, 2.0]
true_bias = 1.5
y_reg = X_reg @ true_coef + true_bias + rng.normal(0, 1.0, size=n2)

np.savetxt("regression_X.csv", X_reg, delimiter=",", fmt="%.8f")
np.savetxt("regression_y.csv", y_reg, delimiter=",", fmt="%.8f")
np.savetxt("regression_true_coef.csv", true_coef, delimiter=",", fmt="%.8f")

print("classification:", X_cls.shape, y_cls.shape, "positive rate:", y_cls.mean())
print("regression:", X_reg.shape, y_reg.shape)
print("true coef:", true_coef, "true bias:", true_bias)
