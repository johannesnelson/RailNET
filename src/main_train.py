from train import train
import model_maker as mm
from custom_dataset import Virginia_Rail_Dataset
import prep_transforms as pt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR



model = mm.create_modified_resnet18(2)

preprocessing_transforms = pt.get_preprocessing_transforms()
augmentation_transforms = pt.get_heavy_augmentation_transforms()
augmentation_noise_transforms = pt.get_heavy_augmentation_noise_transforms()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Virginia_Rail_Dataset(root_dir='', 
                                      positive_folder='../data/positive_class', 
                                      negative_folder='../data/negative_class', 
                                      preprocessing_transforms=preprocessing_transforms, 
                                      augmentation_transforms=augmentation_transforms,
                                      augmentation_noise_transforms=augmentation_noise_transforms,
                                      device=device,
                                    )
test_dataset = Virginia_Rail_Dataset(root_dir='', 
                                     positive_folder='../data/positive_soundscape', 
                                     negative_folder='../data/negative_soundscape', 
                                     preprocessing_transforms=preprocessing_transforms, 
                                     augmentation_transforms=None,
                                     augmentation_noise_transforms=None,
                                     device=device,
                                 )




train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


num_epochs = 60
starting_LR = 0.001
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = starting_LR)
criterion = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=15, gamma=.5)
checkpoint_path = "test"

train_losses, test_losses, train_accs, test_accs, train_f1s, test_f1s, train_recalls, test_recalls = train(
    model,
    train_loader,
    test_loader,
    num_epochs,
    optimizer,
    criterion,
    device,
    scheduler,
    checkpoint_path,
    resume=False,
    manual_lr=None
)


