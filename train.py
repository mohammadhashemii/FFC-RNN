from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import visualize_samples
from dataset import SadriDataset

train_dataset = SadriDataset(root_dir='data', img_size=(256, 32), is_training_set=True)
test_dataset = SadriDataset(root_dir='data', img_size=(256, 32), is_training_set=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#visualize_samples(train_dataset, train_dataset.wv.num_to_word,
#                  n_samples=16, cols=4, random_img=True)

#plt.imshtow(sample['image'].permute(1, 2, 0), cmap='gray')
#plt.show()