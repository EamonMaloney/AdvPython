import numpy as np
from assignment2.assignment2.scripts.nb import build_dataframe, train_nb



train_df, test_df = build_dataframe(r"C:\Users\eamon\AdvPython Coding\assignment2\assignment2\data")

vocabulary, priors, likelihoods = train_nb(train_df)
print("Priors:", priors)
print("Likelihoods shape:", likelihoods.shape)

for a in [0.01, 0.1, 1, 10]:
    _, _, likelihoods_a = train_nb(train_df, alpha=a)
    print(f"alpha={a}: min={likelihoods_a.min():.6f}, max={likelihoods_a.max():.6f}")

preds = test(test_df, vocabulary, priors, likelihoods)
print("My predictions:", preds)

sklearn_preds = sklearn_nb(train_df, test_df)
print("Sklearn predictions:", sklearn_preds)

acc, f1, conf = get_metrics(test_df["author"], preds)
print("My model - accuracy:", acc, "f1:", f1)
print("My model - confusion matrix:\n", conf)

sk_acc, sk_f1, sk_conf = get_metrics(test_df["author"], sklearn_preds)
print("Sklearn - accuracy:", sk_acc, "f1:", sk_f1)
print("Sklearn - confusion matrix:\n", sk_conf)


# Document length by author
train_nb["length"] = train_nb["text"].apply(lambda t: len(t.split()))
print(train_nb.groupby("author")["length"].describe())

# Most common words per author
from collections import Counter
for c in train_nb["author"].unique():
    words = " ".join(train_nb[train_nb["author"] == c]["text"]).split()
    print(f"Class {c} most common:", Counter(words).most_common(15))

# Beginnings/endings
for c in train_df["author"].unique():
    print(f"\nClass {c} sample openings:")
    for t in train_nb[train_nb["author"] == c]["text"].head(3):
        print(" -", t[:60])

# Build the dataframe
train_df, test_df = build_dataframe(r"C:\Users\eamon\AdvPython Coding\assignment2\assignment2\data")
print("train_df shape:", train_df.shape)

# Run train_nb
vocabulary, priors, likelihoods = train_nb(train_df, alpha=0.1)

# --- Check 1: shapes ---
print("\n--- Shape checks ---")
n_classes = train_df["author"].nunique()
print("vocabulary size:", len(vocabulary))
print("priors shape:", priors.shape, "-> expect (n_classes,) =", (n_classes,))
print("likelihoods shape:", likelihoods.shape, "-> expect (n_classes, len(vocabulary)) =", (n_classes, len(vocabulary)))
assert priors.shape == (n_classes,), "priors shape mismatch!"
assert likelihoods.shape == (n_classes, len(vocabulary)), "likelihoods shape mismatch!"

# --- Check 2: priors sum to 1 ---
print("\n--- Priors ---")
print("priors:", priors)
print("sum:", priors.sum(), "-> should be close to 1.0")
assert np.isclose(priors.sum(), 1.0), "priors don't sum to 1!"

# --- Check 3: each class's likelihood row sums to ~1 ---
print("\n--- Likelihoods sum per class ---")
for c in range(n_classes):
    row_sum = likelihoods[c].sum()
    print(f"class {c}: sum = {row_sum} (should be close to 1.0)")
    assert np.isclose(row_sum, 1.0, atol=1e-6), f"class {c} likelihoods don't sum to 1!"

# --- Check 4: no likelihood is exactly zero (smoothing worked) ---
print("\n--- Zero-probability check ---")
num_zeros = np.sum(likelihoods == 0)
print("number of exact-zero entries in likelihoods:", num_zeros, "-> should be 0")
assert num_zeros == 0, "found zero-probability entries -- smoothing isn't being applied!"

# --- Check 5: a word that's rare/absent in one class should still be nonzero there ---
print("\n--- Spot-check a rare word ---")
sample_word = list(vocabulary.keys())[0]  # just grab some word to inspect
idx = vocabulary[sample_word]
print(f"word='{sample_word}' likelihoods across classes:", likelihoods[:, idx])

print("\nAll checks passed!" )
