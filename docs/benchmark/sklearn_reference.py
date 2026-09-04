import numpy as np, time, json
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

X_cls = np.loadtxt("classification_X.csv", delimiter=",")
y_cls = np.loadtxt("classification_y.csv", delimiter=",").astype(int)
X_reg = np.loadtxt("regression_X.csv", delimiter=",")
y_reg = np.loadtxt("regression_y.csv", delimiter=",")

results = {}

# --- Logistic Regression ---
# No L2 penalty so it's directly comparable to our from-scratch version
t0 = time.perf_counter()
clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10)
clf.fit(X_cls, y_cls)
t1 = time.perf_counter()
pred = clf.predict(X_cls)
acc = accuracy_score(y_cls, pred)
results["logistic_regression"] = {
    "weights": clf.coef_.flatten().tolist(),
    "bias": float(clf.intercept_[0]),
    "accuracy": float(acc),
    "fit_seconds": t1 - t0,
}

# --- Ridge ---
alpha = 1.0
t0 = time.perf_counter()
ridge = Ridge(alpha=alpha, solver="cholesky")
ridge.fit(X_reg, y_reg)
t1 = time.perf_counter()
ridge_pred = ridge.predict(X_reg)
results["ridge"] = {
    "alpha": alpha,
    "weights": ridge.coef_.tolist(),
    "bias": float(ridge.intercept_),
    "mse": float(mean_squared_error(y_reg, ridge_pred)),
    "fit_seconds": t1 - t0,
}

# --- Lasso ---
alpha_lasso = 0.1
t0 = time.perf_counter()
lasso = Lasso(alpha=alpha_lasso, max_iter=5000, tol=1e-8)
lasso.fit(X_reg, y_reg)
t1 = time.perf_counter()
lasso_pred = lasso.predict(X_reg)
results["lasso"] = {
    "alpha": alpha_lasso,
    "weights": lasso.coef_.tolist(),
    "bias": float(lasso.intercept_),
    "mse": float(mean_squared_error(y_reg, lasso_pred)),
    "n_nonzero": int(np.sum(np.abs(lasso.coef_) > 1e-6)),
    "fit_seconds": t1 - t0,
}

with open("sklearn_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
