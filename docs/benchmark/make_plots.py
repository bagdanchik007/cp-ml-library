import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X_cls = np.loadtxt("classification_X.csv", delimiter=",")
y_cls = np.loadtxt("classification_y.csv", delimiter=",").astype(int)

# --- Plot 1: Logistic Regression decision boundary ---
w = np.array([4.661143127856667, -4.0906858590342585])
b = 0.45020098891079113

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_cls[y_cls == 0, 0], X_cls[y_cls == 0, 1], c="#3b82f6", label="Klasse 0", alpha=0.7, edgecolor="k", linewidth=0.3)
ax.scatter(X_cls[y_cls == 1, 0], X_cls[y_cls == 1, 1], c="#ef4444", label="Klasse 1", alpha=0.7, edgecolor="k", linewidth=0.3)

x1_range = np.linspace(X_cls[:, 0].min() - 0.5, X_cls[:, 0].max() + 0.5, 100)
x2_boundary = -(w[0] * x1_range + b) / w[1]
ax.plot(x1_range, x2_boundary, "k--", linewidth=2, label="Entscheidungsgrenze (w·x + b = 0)")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Logistic Regression: Entscheidungsgrenze\n(C++-Implementierung, identisch zu sklearn)")
ax.legend()
fig.tight_layout()
fig.savefig("logistic_regression_boundary.png", dpi=140)
plt.close(fig)

# --- Plot 2: Ridge vs Lasso coefficient comparison ---
true_coef = np.loadtxt("regression_true_coef.csv", delimiter=",")
ridge_coef = np.array([3.9214692771209614, 0.08617496953928051, -0.12650358110650683, -2.9535828721295516,
                        0.024632181314768273, -0.09345418307989739, -0.022720975833509113, 2.074508044070821,
                        -0.045378382458846275, 0.014235425271061275])
lasso_coef = np.array([3.789723468178537, 0.0, -0.018944754079223688, -2.858201421536198,
                        0.0, -0.02564731565873673, -0.0, 1.9713861770581609, -0.0, 0.0])

x = np.arange(len(true_coef))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, true_coef, width, label="Wahre Koeffizienten", color="#22c55e")
ax.bar(x, ridge_coef, width, label="Ridge (L2)", color="#3b82f6")
ax.bar(x + width, lasso_coef, width, label="Lasso (L1)", color="#ef4444")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"w{i}" for i in range(len(true_coef))])
ax.set_ylabel("Koeffizientenwert")
ax.set_title("Ridge vs. Lasso: Koeffizienten im Vergleich zur wahren Struktur\n(nur w0, w3, w7 sind wirklich informativ)")
ax.legend()
fig.tight_layout()
fig.savefig("ridge_vs_lasso_coefficients.png", dpi=140)
plt.close(fig)

print("Plots gespeichert.")
