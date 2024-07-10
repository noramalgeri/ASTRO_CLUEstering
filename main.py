import OpenFits as of
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

def percentage(sorted_df):
    # === cut out 98% of data ===
    threshold = int(len(sorted_df)*0.98)
    threshold_intensity = sorted_df.iloc[threshold]["Intensity"]
    sorted_df = sorted_df.drop(df[df["Intensity"]<threshold_intensity].index)
    return sorted_df

def derivative(sorted_df):
    # === calculating derivative ===
    gap = 1000
    sorted_df.reset_index(inplace=True, drop=True)
    derivatives = [0] * len(sorted_df["Intensity"])

    for k in range(len(sorted_df["Intensity"])-gap):
        dfx = (sorted_df["Intensity"][k+gap] - sorted_df["Intensity"][k])/gap
        derivatives[k]=dfx

    sorted_df = sorted_df.assign(derivative = derivatives)
    # print(sorted_df.head)

    # === plot of derivative ===
    plt.scatter(range(len(sorted_df["Intensity"])), derivatives, s=0.01)
    plt.show()

    # === threshold for derivative ===
    threshold = sorted_df[sorted_df.derivative > 0.05].index[0]
    # print(threshold)
    threshold_intensity = sorted_df.iloc[threshold].derivative
    sorted_df.reset_index(inplace=True, drop=True)
    # print(threshold_intensity)

    # === different approaches of dropping data
    # cutoff = sorted_df.drop(sorted_df[sorted_df["Intensity"]<threshold_intensity].index)
    # cutoff = sorted_df.drop(sorted_df[sorted_df.derivative<threshold_intensity].index)
    # cutoff = sorted_df[(sorted_df.derivative < 4).idxmax():]
    cutoff = sorted_df.drop(sorted_df.index[0:threshold+1])
    # print(sorted_df[0:50])

    return cutoff

file = "m29.fit"

data = of.OpenFits(file)

x = []
y = []
intensity = []
for i in range(len(data)):
    for j in range(len(data[i])):
        x.append(j)
        y.append(i)
        intensity.append(data[i][j])
        # print(i, j, data[i][j])

# plt.hist(intensity, log=True, bins=200)
# plt.show()

df = pd.DataFrame(list(zip(x, y, intensity)), columns=['X', 'Y', 'Intensity'])

sorted_df = df.sort_values(by="Intensity")

# === filter data ===
cutoff = percentage(sorted_df)

# cutoff = derivative(sorted_df)

# === plotting scatter map ===
# min_val, max_val = cutoff["Intensity"].iloc[50], cutoff["Intensity"].iloc[-5]
# cmap = matplotlib.cm.coolwarm
# norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
# color_list = cmap(cutoff["Intensity"])

plt.scatter(cutoff['X'],cutoff["Y"], s=1, c=cutoff["Intensity"], cmap='gist_rainbow', marker='*')
# plt.scatter(cutoff['X'],cutoff["Y"], s=0.001)
plt.colorbar()
plt.show()