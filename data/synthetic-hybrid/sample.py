import json

data = json.load(open("/home/krypticmouse/Desktop/Federated-Learning-PyTorch/data/synthetic-hybrid/data/mytrain.json", "r"))

l = []
users = data["users"]
for i in users:
    l.append([int(k) for k in data["user_data"][i]["y"]])

from collections import Counter
import matplotlib.pyplot as plt

for index, i in enumerate(l):
    frequency = Counter(i)
    print(frequency)

    plt.bar(frequency.keys(), frequency.values())
    plt.savefig(f'/home/krypticmouse/Desktop/Federated-Learning-PyTorch/data/synthetic-hybrid/plots/frequency_plot_{index}.png')



