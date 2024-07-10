import OpenFits as of
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import CLUEstering as clue

def percentage(sorted_df):
    # === cut out 98% of data ===
    threshold = int(len(sorted_df)*0.99)
    threshold_intensity = sorted_df.iloc[threshold]["weight"]
    sorted_df = sorted_df.drop(df[df["weight"]<threshold_intensity].index)
    return sorted_df

def derivative(sorted_df):
    # === calculating derivative ===
    gap = 1000
    sorted_df.reset_index(inplace=True, drop=True)
    derivatives = [0] * len(sorted_df["weight"])

    for k in range(len(sorted_df["weight"])-gap):
        dfx = (sorted_df["weight"][k+gap] - sorted_df["weight"][k])/gap
        derivatives[k]=dfx

    sorted_df = sorted_df.assign(derivative = derivatives)
    # print(sorted_df.head)

    # === plot of derivative ===
    plt.scatter(range(len(sorted_df["weight"])), derivatives, s=0.01)
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

df = pd.DataFrame(list(zip(x, y, intensity)), columns=['x0', 'x1', 'weight'])

sorted_df = df.sort_values(by="weight")

# === filter data ===
cutoff = percentage(sorted_df)

dc = 0.0000000000000000000000000000000000000000005 #the side of the box inside of which the density of a point is calculated
rhoc = 500000 #the minimum energy density that a point must have to not be considered an outlier
odf = 0.005 # Outlier Delta Factor: multiplied by dc_ gives dm_, the side of the box inside of which the followers of a point are searched

clust = clue.clusterer(dc, rhoc, odf)
clust.read_data(cutoff)
clust.run_clue()
clust.cluster_plotter()

# cutoff = derivative(sorted_df)

# === plotting scatter map ===
# min_val, max_val = cutoff["Intensity"].iloc[50], cutoff["Intensity"].iloc[-5]
# cmap = matplotlib.cm.coolwarm
# norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
# color_list = cmap(cutoff["Intensity"])

# plt.scatter(cutoff['X'],cutoff["Y"], s=1, c=cutoff["Intensity"], cmap='gist_rainbow', marker='*')
# # plt.scatter(cutoff['X'],cutoff["Y"], s=0.001)
# plt.colorbar()
# plt.show()