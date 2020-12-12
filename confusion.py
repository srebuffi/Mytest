
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product


# cm = confusion_matrix(targets, preds)
cm = confusion_matrix(targets[probs > 0.9], preds[probs > 0.9])
include_values = True

fig, ax = plt.subplots()
n_classes = cm.shape[0]
cmap = 'viridis'
cmap = plt.cm.Blues
im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
text_ = None
cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

if include_values:
    text_ = np.empty_like(cm, dtype=object)
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_cm = format(cm[i, j], '.2g')
        if cm.dtype.kind != 'f':
            text_d = format(cm[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d

        text_[i, j] = ax.text(
            j, i, text_cm,
            ha="center", va="center",
            color=color)

display_labels = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig.colorbar(im_, ax=ax)
ax.set(#xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       # xticklabels=display_labels,
       yticklabels=display_labels,
       ylabel="True label",
       xlabel="Predicted label")
ax.set_xticks([], [])
ax.set_ylim((n_classes - 0.5, -0.5))
plt.setp(ax.get_xticklabels(), rotation='horizontal')
# plt.savefig('confusion_matrix_all.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
plt.savefig('confusion_matrix_confident.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
plt.clf()
percentage_sample_confident = 100. * cm.sum() / 60000  # 59.9
acc_confident = 100. * np.diag(cm).sum() / cm.sum() # 94.0
