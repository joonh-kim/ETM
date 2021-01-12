import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GTA5, Cross-city
data = np.array([[-5.65, -8.32, -5.93, -1.86, -1.79, -1.96, -1.81, -3.53, -4.70, -1.23, 0.86],
                 [-2.01, -3.68, -2.78, -1.71, -0.38, -2.90, -0.18, -1.64, -1.12, -0.54, 0.31],
                 [0.09, -2.86, -1.72, -1.19, -0.07, -1.33, -0.06, -1.01, -1.16, -0.07, 0.83],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

targets = ['Rio', 'Rome', 'Taipei', 'Tokyo (current)']
methods = ['FCNs in the Wild (H)', 'AdaptSegNet (H)', 'AdvEnt (H)', 'SIM (H)', 'CUDA Square (H)',
           'FCNs in the Wild (L)', 'AdaptSegNet (L)', 'AdvEnt (L)', 'ACE (L)', 'SIM (L)', 'CUDA Square (L)']
df = pd.DataFrame(data, targets, methods)

linestyles = ['-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--']
colors = ['g', 'b', 'c', 'm', 'r', 'g', 'b', 'c', 'y', 'm', 'r']

for i, (linestyle, color) in enumerate(zip(linestyles, colors)):
    plt.plot(df.index, df.values[:, i], linestyle=linestyle, color=color)

plt.title('From GTA5')
plt.ylabel('Forgetting')
plt.xlabel('Target Domains')
plt.legend(methods, loc='lower right')
plt.grid(True, axis='y')

plt.show()

# SYNTHIA, Cross-city
data = np.array([[-4.79, -2.56, -2.04, -1.02, 0.16, -1.83, -2.64, -2.50, -6.53, -1.04, 0.30],
                 [-0.34, -1.69, -1.47, -0.68, 0.01, 0.12, -0.08, 0.11, -2.69, -1.49, 1.19],
                 [-1.55, -0.44, -0.25, -0.53, 0.07, -0.08, -0.38, -0.04, -0.43, -0.43, 0.15],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

targets = ['Rio', 'Rome', 'Taipei', 'Tokyo (current)']
methods = ['FCNs in the Wild (H)', 'AdaptSegNet (H)', 'AdvEnt (H)', 'SIM (H)', 'CUDA Square (H)',
           'FCNs in the Wild (L)', 'AdaptSegNet (L)', 'AdvEnt (L)', 'ACE (L)', 'SIM (L)', 'CUDA Square (L)']
df = pd.DataFrame(data, targets, methods)

linestyles = ['-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--']
colors = ['g', 'b', 'c', 'm', 'r', 'g', 'b', 'c', 'y', 'm', 'r']

for i, (linestyle, color) in enumerate(zip(linestyles, colors)):
    plt.plot(df.index, df.values[:, i], linestyle=linestyle, color=color)

plt.title('From SYNTHIA')
plt.ylabel('Forgetting')
plt.xlabel('Target Domains')
plt.legend(methods, loc='lower right')
plt.grid(True, axis='y')

plt.show()

# GTA5, CityScapes-IDD
FCNs_L = -5.71
AdaptSegNet_L = -2.42
AdvEnt_L = -1.71
ACE_L = -1.82
SIM_L = -0.71
CUDA_square_L = -1.26

FCNs_H = -3.81
AdaptSegNet_H = -7.41
AdvEnt_H = -5.86
SIM_H = -0.64
CUDA_square_H = -1.35

bar_width = 0.1
x = 0.85
alpha = 0.5

p01 = plt.bar(0, FCNs_L,
             bar_width,
             color='g',
             alpha=alpha,
             label='FCNs in the Wild')

p02 = plt.bar(bar_width, AdaptSegNet_L,
             bar_width,
             color='b',
             alpha=alpha,
             label='AdaptSegNet')

p03 = plt.bar(2 * bar_width, AdvEnt_L,
             bar_width,
             color='c',
             alpha=alpha,
             label='AdvEnt')

p04 = plt.bar(3 * bar_width, ACE_L,
             bar_width,
             color='y',
             alpha=alpha,
             label='ACE')

p05 = plt.bar(4 * bar_width, SIM_L,
             bar_width,
             color='m',
             alpha=alpha,
             label='SIM')

p06 = plt.bar(5 * bar_width, CUDA_square_L,
             bar_width,
             color='r',
             alpha=alpha,
             label='CUDA Square')

p11 = plt.bar(x, FCNs_H,
             bar_width,
             color='g',
             alpha=alpha,
             label='FCNs in the Wild')

p12 = plt.bar(x + bar_width, AdaptSegNet_H,
             bar_width,
             color='b',
             alpha=alpha,
             label='AdaptSegNet')

p13 = plt.bar(x + 2 * bar_width, AdvEnt_H,
             bar_width,
             color='c',
             alpha=alpha,
             label='AdvEnt')

p14 = plt.bar(x + 3 * bar_width, SIM_H,
             bar_width,
             color='m',
             alpha=alpha,
             label='SIM')

p15 = plt.bar(x + 4 * bar_width, CUDA_square_H,
             bar_width,
             color='r',
             alpha=alpha,
             label='CUDA Square')

plt.title('From GTA5')
plt.ylabel('Forgetting')
plt.xlabel('Input Size')
plt.xticks([2.5 * bar_width, x + 2 * bar_width], ["L", "H"])
plt.legend((p01[0], p02[0], p03[0], p04[0], p05[0], p06[0]),
           ('FCNs in the Wild', 'AdaptSegNet', 'AdvEnt', 'ACE', 'SIM', 'CUDA Square'), loc='lower center')
plt.show()

# SYNTHIA, CityScapes-IDD
FCNs_L = 0.71
AdaptSegNet_L = -0.08
AdvEnt_L = -0.26
ACE_L = -1.14
SIM_L = -0.04
CUDA_square_L = 0.75

FCNs_H = -2.27
AdaptSegNet_H = -3.33
AdvEnt_H = -6.39
SIM_H = -1.35
CUDA_square_H = -1.92

bar_width = 0.1
x = 0.85
alpha = 0.5

p01 = plt.bar(0, FCNs_L,
             bar_width,
             color='g',
             alpha=alpha,
             label='FCNs in the Wild')

p02 = plt.bar(bar_width, AdaptSegNet_L,
             bar_width,
             color='b',
             alpha=alpha,
             label='AdaptSegNet')

p03 = plt.bar(2 * bar_width, AdvEnt_L,
             bar_width,
             color='c',
             alpha=alpha,
             label='AdvEnt')

p04 = plt.bar(3 * bar_width, ACE_L,
             bar_width,
             color='y',
             alpha=alpha,
             label='ACE')

p05 = plt.bar(4 * bar_width, SIM_L,
             bar_width,
             color='m',
             alpha=alpha,
             label='SIM')

p06 = plt.bar(5 * bar_width, CUDA_square_L,
             bar_width,
             color='r',
             alpha=alpha,
             label='CUDA Square')

p11 = plt.bar(x, FCNs_H,
             bar_width,
             color='g',
             alpha=alpha,
             label='FCNs in the Wild')

p12 = plt.bar(x + bar_width, AdaptSegNet_H,
             bar_width,
             color='b',
             alpha=alpha,
             label='AdaptSegNet')

p13 = plt.bar(x + 2 * bar_width, AdvEnt_H,
             bar_width,
             color='c',
             alpha=alpha,
             label='AdvEnt')

p14 = plt.bar(x + 3 * bar_width, SIM_H,
             bar_width,
             color='m',
             alpha=alpha,
             label='SIM')

p15 = plt.bar(x + 4 * bar_width, CUDA_square_H,
             bar_width,
             color='r',
             alpha=alpha,
             label='CUDA Square')

plt.title('From SYNTHIA')
plt.ylabel('Forgetting')
plt.xlabel('Input Size')
plt.xticks([2.5 * bar_width, x + 2 * bar_width], ["L", "H"])
plt.legend((p01[0], p02[0], p03[0], p04[0], p05[0], p06[0]),
           ('FCNs in the Wild', 'AdaptSegNet', 'AdvEnt', 'ACE', 'SIM', 'CUDA Square'), loc='lower center')
plt.show()

