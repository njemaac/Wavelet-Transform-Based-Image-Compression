import pandas as pd
import matplotlib.pyplot as plt

# Učitaj CSV
df = pd.read_csv("results.csv")

# Očisti prazne redove ako ih ima
df = df.dropna()

# ===== AVERAGE =====
df_avg = df.groupby(["method", "size", "threshold"])["timeMs"].mean().reset_index()

# =========================
# 📊 GRAPH 1: SIZE TEST
# =========================
fixed_threshold = df_avg[df_avg["threshold"] == 20]

plt.figure()

for method in fixed_threshold["method"].unique():
    data = fixed_threshold[fixed_threshold["method"] == method]
    plt.plot(data["size"], data["timeMs"], marker='o', label=method)

plt.xlabel("Image Size")
plt.ylabel("Avg Time (ms)")
plt.title("Performance vs Image Size (Threshold = 20%)")
plt.legend()
plt.grid()

plt.savefig("size_test.png")
plt.show()


# =========================
# 📊 GRAPH 2: THRESHOLD TEST
# =========================
fixed_size = df_avg[df_avg["size"] == 1024]

plt.figure()

for method in fixed_size["method"].unique():
    data = fixed_size[fixed_size["method"] == method]
    plt.plot(data["threshold"], data["timeMs"], marker='o', label=method)

plt.xlabel("Threshold (%)")
plt.ylabel("Avg Time (ms)")
plt.title("Performance vs Compression Level (Size = 1024)")
plt.legend()
plt.grid()

plt.savefig("threshold_test.png")
plt.show()
