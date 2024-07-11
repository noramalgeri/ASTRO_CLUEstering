import OpenFits as of
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import CLUEstering as clue
import pprint
import numpy as np

def percentage(sorted_df, percentile):
    # === cut out 98% of data ===
    threshold = int(len(sorted_df)*percentile)
    threshold_intensity = sorted_df.iloc[threshold]["weight"]
    print(threshold_intensity)
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

data = of.OpenFits(file, BoW = 0)




x = []
y = []
intensity = []
for i in range(len(data)):
    for j in range(len(data[i])):
        x.append(j)
        y.append(i)
        intensity.append(data[i][j])
#        print(i, j, data[i][j])

# plt.hist(intensity, log=True, bins=200)
# plt.show()






floatx = np.array(x,dtype=float)
floaty = np.array(y,dtype=float)
intensity_np = np.array(intensity)

df = pd.DataFrame({'x0':floatx, 'x1':floaty, 'weight':intensity})

sorted_df = df.sort_values(by="weight")

# # === filter data ===
cutoff = percentage(sorted_df, 0.96)
# plt.scatter(cutoff['x0'],cutoff["x1"], s=1, c=cutoff["weight"], cmap='gist_rainbow', marker='*')

absolute_dc = 3
scaling = np.std(x)

critical_energy_density = 0.35*cutoff.iloc[0]["weight"]*scaling*scaling



print(scaling)

## === CLUE algorithm ===
dc = absolute_dc/scaling #the side of the box inside of which the density of a point is calculated
rhoc = critical_energy_density*dc*dc #the minimum energy density that a point must have to not be considered an outlier
odf = 1 # Outlier Delta Factor: multiplied by dc_ gives dm_, the side of the box inside of which the followers of a point are searched
# # ppb = 1000

print("Running CLUE with ", dc, rhoc, odf)
cutoff.sort_index(inplace=True)

print("Number of points in dataframe:", len(cutoff))

clust = clue.clusterer(dc, rhoc, odf)
clust.read_data(cutoff)
clust.run_clue()
clust.cluster_plotter()



# print(clust.n_clusters)
# print(clust.coords)
# print(len(clust.is_seed))
# print(clust.clusters)
print("Number of outliers: ", len(clust.cluster_points[-1]))

noise_list = []
for cluster in clust.cluster_points:
    if len(cluster) < 20 or clust.cluster_ids[cluster[0]] == -1:
        for point in cluster:
            noise_list.append(cutoff.iloc[point]["x0"])

plt.hist(noise_list, log=False, bins=200)
plt.show()



# for point in clust.cluster_points[1]:
#     print(cutoff.iloc[point]["x0"])
# print(clust.cluster_points[1])

# === print seed histogram ===
# outlier_list = []
# k = 0
# for row, target in zip(cutoff.iterrows(), clust.is_seed):
#     if target == 1:
#         outlier_list.append(row[1]["x0"])
# print(len(outlier_list))
# plt.hist(outlier_list, log=False, bins=200)
# plt.show()

# print(outlier_or_seed[0])
# print(max(df['x0']))

# print( and cutoff.outlier_or_seed == 1])
# temp = cutoff[cutoff["x0"] == 1]
# print(len(temp[temp.outlier_or_seed == 1]))

# === append to dataframe whether point is a (outlier, seed) or not ===
# cutoff = cutoff.assign(outlier_or_seed = clust.is_seed)
# outliers_over_x_axis = []
# for i in range(0, int(max(df['x0']))):
#     temp = cutoff[cutoff["x0"] == i]
#     outliers_over_x_axis.append(len(temp[temp.outlier_or_seed == 1]))
# print(len(outliers_over_x_axis))

# plt.scatter(range(0, int(max(df['x0']))), outliers_over_x_axis)
# plt.show()

# cutoff = derivative(sorted_df)

# === plotting scatter map ===
# min_val, max_val = cutoff["Intensity"].iloc[50], cutoff["Intensity"].iloc[-5]
# cmap = matplotlib.cm.coolwarm
# norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
# color_list = cmap(cutoff["Intensity"])

# plt.scatter(cutoff['x0'],cutoff["x1"], s=1, c=cutoff["weight"], cmap='gist_rainbow', marker='*')
# plt.scatter(cutoff['X'],cutoff["Y"], s=0.001)
# plt.colorbar()
# plt.show()
