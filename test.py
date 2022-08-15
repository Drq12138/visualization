import numpy as np

acc_file = '/home/DiskB/rqding/checkpoints/visualization/adam_lr_0.001_decay_0.0001/save_net_resnet32_orig_acc.npy'

loss_file = '/home/DiskB/rqding/checkpoints/visualization/adam_lr_0.001_decay_0.0001/save_net_resnet32_orig_loss.npy'

acc = np.load(acc_file)
loss = np.load(loss_file)

pass