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


def cut_corners(df, x1, y1, x2, y2, corner):
    # Calculate the slope (m) and intercept (c) of the line y = mx + c
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
    else:
        # Vertical line case
        m = float('inf')
        c = None

    # Determine if each point (x0, y0) is above the line
    if m != float('inf'):
        if corner == "top left":
            df.loc[df['x1'] > m * df['x0'] + c, 'weight'] = 0
        if corner == "top right":
            df.loc[df['x1'] > m * df['x0'] + c, 'weight'] = 0
        if corner == "bottom left":
            df.loc[df['x1'] < m * df['x0'] + c, 'weight'] = 0
        if corner == "bottom right":
            df.loc[df['x1'] < m * df['x0'] + c, 'weight'] = 0
    else:
        # For vertical line case
        df.loc[df['x0'] > x1, 'weight'] = 0

    return df

# ==================== reading data =====================

file = "m29.fit"
file_bkg = "atlas_pleione_2s_000_201_bkg.fit"

data = of.OpenFits(file, overlay=True)
data_bkg = of.OpenFits(file_bkg, overlay=False)

# ======================== setting data =================

x = []
y = []
intensity = []
intensity_b = []
for i in range(len(data)):
    for j in range(len(data[i])):
        x.append(j)
        y.append(i)
        intensity.append(data[i][j])
        intensity_b.append(data_bkg[i][j])
#        print(i, j, data[i][j])

# plt.hist(intensity, log=True, bins=200)
# plt.show()

floatx = np.array(x,dtype=float)
floaty = np.array(y,dtype=float)
intensity_np = np.array(intensity)

df = pd.DataFrame({'x0':floatx, 'x1':floaty, 'old_weight':intensity, 'background':intensity_b})

#================== cutting corners and removing gradient ===========================

df_sub = df['old_weight'].subtract(df['background'])
# print(type(df_sub))
# df.loc[:, "weight"] = df_sub[]
df['weight'] = df_sub

df = cut_corners(df, 0, 937, 130, 1000, "top left")
df = cut_corners(df, 0, 226, 230, 0, "bottom left")
df = cut_corners(df, 1420, 1020, 1530, 970, "top right")


#=============== removing lower intensities ============================

sorted_df = df.sort_values(by="weight")

# # === filter data ===
cutoff = percentage(sorted_df, 0.98)
# plt.scatter(cutoff['x0'],cutoff["x1"], s=1, c=cutoff["weight"], cmap='gist_rainbow', marker='*')
# plt.show()
# print(df)

# plt.scatter(df['x0'],df["x1"], s=1, c=df["weight"], cmap='gist_rainbow', marker='*')
# plt.show()

# ==================== CLUE algorithm ==============================

absolute_dc = 3
scaling = np.std(x)
# print(scaling)
critical_energy_density = 0.35*cutoff.iloc[0]["weight"]*scaling*scaling
dc = absolute_dc/scaling #the side of the box inside of which the density of a point is calculated
rhoc = critical_energy_density*dc*dc #the minimum energy density that a point must have to not be considered an outlier
odf = 1 # Outlier Delta Factor: multiplied by dc_ gives dm_, the side of the box inside of which the followers of a point are searched
# # ppb = 1000

print("Running CLUE with ", dc, rhoc, odf)
cutoff.sort_index(inplace=True)

print("Number of points in dataframe:", len(cutoff))

clust = clue.clusterer(dc, rhoc, odf)
clust.read_data(cutoff.drop(['old_weight', 'background'], axis=1))
clust.run_clue()

# ============== cluster stats ===============================

# print(clust.n_clusters)
# print(clust.coords)
# print("Lenght of is_seed array", len(clust.is_seed))
# print("Lenght of clust.cluster_points", len(clust.cluster_points))
# print("Lenght of cluster_ids", len(clust.cluster_ids))
# print(clust.clusters)
# print("Number of outliers: ", len(clust.cluster_points[-1]))
# print("Before", len(clust.cluster_points))
# print(type(clust.cluster_points))


# ============== post processing ===========================

for cluster in clust.cluster_points:
    if len(cluster) < 10 or clust.cluster_ids[cluster[0]] == -1:
        for point in cluster:
            if clust.is_seed[point] == 1:
                clust.is_seed[point] = 0
            clust.cluster_ids[point] = -1

clust.cluster_plotter()

# ============== trial and error ===========================

# noise_list = []
# for cluster in clust.cluster_points:
#     # print(type(cluster))
#     if len(cluster) < 20 or clust.cluster_ids[cluster[0]] == -1:
        # clust.cluster_points.remove(cluster)
        # np.delete(clust.cluster_points, [clust.cluster_ids[cluster]])
        # print("Removed")
        # print(cluster)
        # for point in cluster:
        #     if clust.is_seed[point] == 1:
        #         clust.cluster_ids[point] = -1
        #         clust.is_seed[point] = 0
            # noise_list.append(cutoff.iloc[point]["weight"])
            # cluster.remove(point)
            # cutoff.iloc[point]["weight"] = 0

# plt.scatter(cutoff['x0'], cutoff["x1"], s=1, c=cutoff["weight"], cmap='gist_rainbow', marker='*')
# plt.show()

# plt.hist(noise_list, log=False, bins=200)
# plt.show()

# =========================== unused code ===================================

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


