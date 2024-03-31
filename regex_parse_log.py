
#%%
import re
import pandas as pd
import matplotlib.pyplot as plt

# We want to extract the epoch number, the loss, the training loss, and the losses for each environment.

## Read the file log.txt file
# run_folder = "exp/LongRun/"
run_folder = "exp/2024-03-31 11:56:42.500772/"
with open(run_folder+'log.txt', 'r') as file:
    log = file.readlines()

#%%
## Here's an example of a line from a file that we want to parse:
# [1][1/1200][1/1] | loss: 9.959e-02 | loss_train: 1.091e-02 | loss_e0: 9.579e-02 | loss_e1: 5.455e-02 | loss_e2: 1.484e-01

## Collect the epochs numbers that are followed by loss
epochs = re.findall(r'\[(\d+)\]\[(\d+)/(\d+)\]\[(\d+)/(\d+)\] \| loss:', ' '.join(log))
epochs = [int(epoch[0]) for epoch in epochs]

## Collect the loss values
losses = re.findall(r'loss: ([\d.e(-|+)]+)', ' '.join(log))

## Collect the training loss values
train_losses = re.findall(r'loss_train: ([\d.e(-|+)]+)', ' '.join(log))

## Collect the environment losses individually
env_losses_0 = re.findall(r'loss_e0: ([\d.e(-|+)]+)', ' '.join(log))
env_losses_1 = re.findall(r'loss_e1: ([\d.e(-|+)]+)', ' '.join(log))
env_losses_2 = re.findall(r'loss_e2: ([\d.e(-|+)]+)', ' '.join(log))

## Put these in a dataframe in floats
df = pd.DataFrame({'Epoch': epochs, 'Loss': losses, 'TrainLoss': train_losses, 'Env0': env_losses_0, 'Env1': env_losses_1, 'Env2': env_losses_2})

## Convert to float if possible, otherwise keep as string
df = df.apply(pd.to_numeric, errors='coerce')   ## new
print(df.head())

## Plot the data from the dataframe
plt.plot(df['Epoch'], df['Loss'], label='Loss')
plt.plot(df['Epoch'], df['TrainLoss'], label='Train Loss')
plt.plot(df['Epoch'], df['Env0'], label='Env Loss 0')
plt.plot(df['Epoch'], df['Env1'], label='Env Loss 1')
plt.plot(df['Epoch'], df['Env2'], label='Env Loss 2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# set limits
plt.yscale('log')
plt.ylim(1e-4, 1e-1)
plt.legend()
plt.show()




#%%

## Everyb now an then, we have the mean loss in a line like this
# [50][50/1200][1/1] | loss_test_mean: 2.819e-02 | loss_test_std: 3.839e-03

## Collect only yhe epochs numbers that are followed by the mean loss
epochs_means = re.findall(r'\[(\d+)\]\[(\d+)/(\d+)\]\[(\d+)/(\d+)\] \| loss_test_mean', ' '.join(log))
epochs_means = [int(epoch[0]) for epoch in epochs_means]

# loss_means = re.findall(r'loss_test_mean: ([\d.e-]+)', ' '.join(log))

## The loss can have a positive or negative exponent, so we need to account for that
loss_means = re.findall(r'loss_test_mean: ([\d.e(-|+)]+)', ' '.join(log))


loss_stds = re.findall(r'loss_test_std: ([\d.e(-|+)]+)', ' '.join(log))

df_means = pd.DataFrame({'Epoch': epochs_means, 'Loss': loss_means, 'Std': loss_stds})
print(df_means.tail())

df_means = df_means.apply(pd.to_numeric, errors='coerce')

print(df_means)

## Plot the data
plt.errorbar(df_means["Epoch"], df_means["Loss"], yerr=df_means["Std"], label='Loss Mean')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()
