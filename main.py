import OpenFits as of

file = "m29.fit"

data = of.OpenFits(file)


for i in range(len(data)):
    for j in range(len(data[i])):
        print(i, j, data[i][j])