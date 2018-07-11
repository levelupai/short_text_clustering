import pickle
import numpy
import matplotlib.pyplot as plt

lines = open('word_ids.py', 'r').readline()
word_to_id = eval(lines)

components = None
with open('components_dump.txt', 'br') as c:
    components = pickle.load(c)

windows_sum = sum(components[0]) # windows
not_windows_sum = sum(components[1]) # not windows



colors = []

area = []
for i in range(len(components[0])):
    area.append(max(components[1][i], components[0][i])*10)  # 0 to 15 point radii

    not_windows_proportion = components[1][i] / not_windows_sum
    windows_proportion = components[0][i] / windows_sum
    if not_windows_proportion == windows_proportion:
        colors.append(0)
    elif not_windows_proportion > windows_proportion:
        colors.append(1)
    else:
        colors.append(2)

labels = ['Unknown', 'Non-Windows', 'Windows']
fig, ax = plt.subplots()
color_choices = ['black', 'blue', 'green']
for i in range(1, 3):
    x_subset = []
    y_subset = []
    area_subset = []
    for j in range(len(colors)):
        if colors[j] == i:
            x_subset.append(components[0][j])
            y_subset.append(components[1][j])
            area_subset.append(area[j])
    with open('dumpfile', 'w+') as w:
        w.write("%s %s %s %s %s %s" % (x_subset, y_subset, area_subset, i, 0.5, labels[i]))
    ax.scatter(x_subset, y_subset, s=area_subset, c=color_choices[i], alpha=0.5, label=labels[i])

ax.legend()
ax.grid(False)
plt.show()
