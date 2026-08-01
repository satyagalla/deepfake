import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier

rng = np.random.RandomState(7)

# Two well-separated "core" clusters -- easy, unambiguous points.
real_core = rng.normal(loc=[-2, 0.5], scale=0.6, size=(14, 2))
fake_core = rng.normal(loc=[2, -0.5], scale=0.6, size=(14, 2))

# Extra "fake" points: still correctly on the fake side, just far away,
# off to the upper-right. These don't sit near the boundary at all --
# they're the "confidently correct, overshooting" points from the explanation.
fake_far = rng.normal(loc=[8, 5], scale=0.7, size=(8, 2))

X = np.vstack([real_core, fake_core, fake_far])
y = np.array([0] * len(real_core) + [1] * (len(fake_core) + len(fake_far)))

log_reg = LogisticRegression().fit(X, y)
ridge = RidgeClassifier().fit(X, y)

def boundary_line(model, xs):
    # RidgeClassifier gives coef_ shape (n_features,) for binary targets;
    # LogisticRegression gives shape (1, n_features). Normalize both to a flat vector.
    w = model.coef_.ravel()
    b = np.ravel(model.intercept_)[0]
    # w0*x + w1*y + b = 0  ->  y = -(w0*x + b) / w1
    return -(w[0] * xs + b) / w[1]

xs = np.linspace(-5, 11, 200)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

for ax, model, title in zip(
    axes, [log_reg, ridge], ["Logistic Regression (log-loss)", "Ridge Classifier (squared error)"]
):
    ax.scatter(real_core[:, 0], real_core[:, 1], c="#3b82f6", label="real (y=0)", edgecolor="k", zorder=3)
    ax.scatter(fake_core[:, 0], fake_core[:, 1], c="#ef4444", label="fake, near boundary (y=1)", edgecolor="k", zorder=3)
    ax.scatter(fake_far[:, 0], fake_far[:, 1], c="#ef4444", marker="^", s=90,
               label="fake, far/confident (y=1)", edgecolor="k", zorder=3)

    preds = model.predict(X)
    misclassified = X[preds != y]
    if len(misclassified):
        ax.scatter(misclassified[:, 0], misclassified[:, 1], facecolors="none",
                   edgecolors="black", s=260, linewidths=2, zorder=4, label="misclassified")

    ax.plot(xs, boundary_line(model, xs), "k--", linewidth=2, label="decision boundary")
    ax.set_title(title)
    ax.set_xlim(-5, 11)
    ax.set_ylim(-3, 8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("feature 1")

axes[0].set_ylabel("feature 2")
axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.suptitle("Same points, same linear model shape, different loss -> different boundary", fontsize=12)
fig.tight_layout()
fig.savefig("loss_boundary_demo.png", dpi=150)

n_mis_log = (log_reg.predict(X) != y).sum()
n_mis_ridge = (ridge.predict(X) != y).sum()
print(f"Logistic regression misclassified: {n_mis_log}")
print(f"Ridge misclassified: {n_mis_ridge}")
print("logreg coef/intercept:", log_reg.coef_, log_reg.intercept_)
print("ridge coef/intercept:", ridge.coef_, ridge.intercept_)
