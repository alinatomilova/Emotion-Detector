import matplotlib.pyplot as plt
import torch
import numpy as np

def create_sequences(data, seq_length, step, is_train = False):
    X, y = [], []

    st = step if is_train else seq_length

    for id in data['userid'].unique():
        user_data = data[data['userid'] == id].sort_values('index')        
        user_X = user_data.drop(columns=['emotionIndex', 'index', 'userid', 'Unnamed: 0']).values
        user_y = user_data['emotionIndex'].values

        if len(user_data) < seq_length + 1:
            continue

        for j in range(0, len(user_data) - seq_length, st):
            X.append(user_X[j:j+seq_length])
            y.append(user_y[j+seq_length])

    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype=np.int64)
    
    return torch.FloatTensor(X), torch.LongTensor(y)


def plot_train_dynamics(model, train_loss, train_f1, title):
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(title, fontweight='bold')

    axs[0].plot(train_loss, color='royalblue')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss dynamics')


    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.detach().numpy().flatten())

    axs[1].plot(train_f1, color='teal')
    axs[1].set_title('F1 score dynamics')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('f1_score')

    return fig