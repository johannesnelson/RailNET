from train import train
import model_maker as mm
from custom_dataset import Virginia_Rail_Dataset
import prep_transforms as pt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR



def train_RailNET(mod_type: str, 
                  augmentation_level: str, 
                  positive_train, 
                  negative_train, 
                  positive_test, 
                  negative_test, 
                  num_epochs, 
                  starting_LR,
                  checkpoint_path):
    
    if mod_type == 'resnet18':
        model = mm.create_modified_resnet18(2)
    if mod_type == 'resnet34':
        model = mm.create_modified_resnet34(2)



    preprocessing_transforms = pt.get_preprocessing_transforms()

    if augmentation_level == 'moderate':
        augmentation_transforms = pt.get_moderate_augmentation_transforms()
        augmentation_noise_transforms = pt.get_moderate_augmentation_noise_transforms()
    if augmentation_level == 'heavy':
        augmentation_transforms = pt.get_heavy_augmentation_transforms()
        augmentation_noise_transforms = pt.get_heavy_augmentation_noise_transforms() 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Virginia_Rail_Dataset(root_dir='', 
                                        positive_folder=positive_train, 
                                        negative_folder=negative_train, 
                                        preprocessing_transforms=preprocessing_transforms, 
                                        augmentation_transforms=augmentation_transforms,
                                        augmentation_noise_transforms=augmentation_noise_transforms,
                                        device=device,
                                        )
    test_dataset = Virginia_Rail_Dataset(root_dir='', 
                                        positive_folder=positive_test, 
                                        negative_folder=negative_test, 
                                        preprocessing_transforms=preprocessing_transforms, 
                                        augmentation_transforms=None,
                                        augmentation_noise_transforms=None,
                                        device=device,
                                    )




    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    num_epochs = num_epochs
    starting_LR = starting_LR
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = starting_LR)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=15, gamma=.5)
    checkpoint_path = checkpoint_path

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

    return train_losses, test_losses, train_accs, test_accs, train_f1s, test_f1s, train_recalls, test_recalls






# Usage example
# train_RailNET(mod_type= 'resnet34',  
#               augmentation_level= 'moderate', 
#               positive_train= "..//data//positive_class", 
#               negative_train= "..//data//negative_class",
#               positive_test='..//data//positive_soundscape',
#               negative_test='..//data//negative_soundscape',
#               num_epochs=70,
#               starting_LR=0.001,
#               checkpoint_path="..//..//ResNet34_moderate_nnc//ResNet34_moderate_nnc.pth")



