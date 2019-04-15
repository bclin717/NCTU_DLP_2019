import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix

results_rs18 = {
    'Test(w/o pretraining)': [73.35231316725978, 73.35231316725978, 73.35231316725978, 73.20996441281139,
                              73.35231316725978,
                              73.35231316725978, 73.35231316725978, 73.35231316725978, 73.35231316725978,
                              73.35231316725978],
    'Test(with pretrining)': [78.26334519572954, 78.71886120996442, 79.27402135231317, 79.87188612099644,
                              77.09608540925267,
                              78.84697508896797, 80.2846975088968, 80.0, 79.35943060498221, 80.14234875444839],
    'Train(w/o pretraining)': [73.35136481725328, 73.50795401971601, 73.50795401971601, 73.50439517420548,
                               73.50795401971601, 73.50795401971601,
                               73.50795401971601, 73.50795401971601, 73.50795401971601, 73.50795401971601],
    'Train(with pretrining)': [74.35140040570839, 77.29100679739493, 78.96366418733763, 79.83914018292467,
                               80.75020463361686,
                               81.10253033915798, 81.94241787963985, 82.18441937435496, 82.74315811950603,
                               83.3446030107833]
}

results_rs50 = {
    'Test(w/o pretraining) ': [68.66903914590748, 73.35231316725978, 73.35231316725978, 71.55871886120997,
                               73.01067615658363],
    'Test(with pretrining)': [77.60854092526691, 78.69039145907473, 78.49110320284697, 79.82918149466192,
                              79.40213523131672],
    'Train(w/o pretraining)': [72.35132922879818, 73.31933520765864, 73.49371863767394, 73.50083632869497,
                               73.50795401971601],
    'Train(with pretrining)': [74.72863802982313, 77.71806825865688, 79.09178262571622, 79.75728673618278,
                               80.42634969215986]
}


def showResult(title='', results=results_rs50):
    plt.figure(title, figsize=(15, 7))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.grid()
    for label, data in results.items():
        plt.plot(range(1, len(data) + 1), data, 'o-' if 'w/o' in label else '-', label=label)
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show()


# showResult(title='Result Comparison(ResNet18)', results=results_rs18)
# showResult(title='Result Comparison(ResNet50)', results=results_rs50)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax
