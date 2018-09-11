import pickle
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = 'results'
MODELS = [
    # ['lenet_big'],
    # ['lenet_dropout_big'],
    # ['lenet_deep_big'],
    # ['lenet_deep1_big'],
    ['lenet_deep2_big'],
    # ['lenet_deep3_big'],
    # ['lenet_deep2_batch_big'],
    # ['resnet_big'],
    ['densenet_big'],
    ['densenet_dropout_big'],
    # ['densenet_batch_after_dropout'],
    # ['densenet_batch_after_no_dropout'],
    ['lstm_64'],
    ['lstm_128'],
    ['cnn_linear_lstm'],
    ['cnn_2lstm'],
    ['cnn_3lstm'],
    # ['densenet_batch_before_no_dropout'],
]
colours = ['b', 'g', 'r', 'y','c','m','k']


val_errors = []
val_stds = []

i = 0
while i < len(MODELS):
    j = 0

    while j < len(MODELS[i]):
        plt.subplot(3,1,1)
        history_data = pickle.load(open(SAVE_DIR + '/' + MODELS[i][j] + '.p', 'rb'))
        loss = history_data['loss_history']
        iters_per_epoch = int(len(loss)/100)
        loss_history = [loss[item] for item in range(0,len(loss),iters_per_epoch)]
        plt.semilogy(loss_history, '{}+-'.format(colours[i]), label=MODELS[i][j])
        plt.legend()

        plt.subplot(3,1,2)
        val_error = history_data['val_error_means']
        plt.plot(np.array(val_error)*255, '{}+-'.format(colours[i]), label=MODELS[i][j])
        plt.legend()

        val_errors.append(np.min(val_error))
        val_std = history_data['val_error_stds']
        val_stds.append(val_std[np.argmin(val_error)])
        j += 1
    i+=1

val_errors = np.array(val_errors)*255
val_stds = np.array(val_stds)*255

plt.legend()
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
x_pos = np.arange(len(MODELS))

ax = plt.subplot(3,1,3)
plt.bar(x_pos, val_errors, yerr=val_stds, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.ylabel('Error')
plt.xticks(x_pos,MODELS)
ax.yaxis.grid(True)

# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
plt.show()
