!pip install ml_collections
!pip install -U neptune-pytorch
!pip install neptune


from models import *
from data_loader import MedicalImageDataset
from losses_metrics import *
from evaluate import *
from trainer import train_model

import neptune
import os
from getpass import getpass
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZjVmMmM4OS1jYjNkLTRlMTAtYjgxYy01MzE0ZjcwMWZiYjgifQ=="
os.environ["NEPTUNE_PROJECT"] = "MedicalImaging/CCA"

run = neptune.init_run()

from torchvision.models.vision_transformer import VisionTransformer



train_set = MedicalImageDataset('train', noise_typ = 'speckle', augment = True, noise = True )



import matplotlib.pyplot as plt
import numpy as np
img, mask = train_set[1]
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow( img.numpy().reshape(256,256,3))
plt.subplot(122)
plt.imshow(mask.numpy().reshape(256,256,1), cmap ='gray')
plt.show()

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--n-channels', type=int, default=3, help='Number of input channels')


    return parser.parse_args()

args = get_args()
args.lr = .0001
args.load = None
args.bilinear = None
args.epochs = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parameters = {
"model_filename": "ViT Full Image",
'Epochs':          {args.epochs},
'Batch size':      {args.batch_size},
'Learning rate':   {args.lr},
'Training size':   {700},
'Validation size': {200},
'Device':          {device.type},
'Images dim':  {'(256,256)'},
'Mixed Precision': {args.amp}
    }
args.run = run

image_size = 256
patch_size = 32
num_layers = 12  # Number of transformer layers
num_heads = 8    # Number of attention heads
hidden_dim = 512  # Hidden dimension of the transformer
mlp_dim = 1024   # Hidden dimension of the MLP (feedforward) layers

if __name__ == '__main__':
    
    # Instantiate VisionTransformer model
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes= 1  # Number of output classes (adjust as per your task)
        )
    #model = Iternet(n_channels=1, n_classes=1, out_channels=32, iterations=2, transformer_config=TRANSCONFIG["ViT-B_32"])
    #model = VisionTransformer(TRANSCONFIG['R50-ViT-B_16'], img_size=256, num_classes=1, zero_head=False, vis=True)
    #model = Iternet( n_classes=1, out_channels=92, iterations=2, transformer_config=TRANSCONFIG["ViT-B_32"])
    #model = UNet_base(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    #model = Metapolyp(in_chans = 1)
    model = model.to(memory_format=torch.channels_last)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    # Use DataParallel to wrap the model

    npt_logger = NeptuneLogger(
        run=run,
        model=model,
        log_model_diagram=True,
        log_gradients=False,
        log_parameters=False,
        log_freq=30,
    )
    run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported( 
    parameters
    )
    # Add Neptune Logger to the Args function
    args.npt_logger = npt_logger
    
    model = nn.DataParallel(model)
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device)

    try:
        train_model(
            model=model,
            args = args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            run = run,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

model = VisionTransformer(TRANSCONFIG['R50-ViT-B_16'], img_size=256, num_classes=1, zero_head=False, vis=True)
model = nn.DataParallel(model)
args.load = '/kaggle/working/checkpoints/best_checkpoint.pth'
state_dict = torch.load(args.load, map_location=device)
del state_dict['mask_values']
model.load_state_dict(state_dict)
logging.info(f'Model loaded from {args.load}')

from torch.utils.data import DataLoader
test_set   = MedicalImageDataset('test',noise_typ=None)
loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
test_loader = DataLoader(test_set, shuffle=True, **loader_args)

criterion = nn.CrossEntropyLoss() if args.classes > 1 else DiceBCELoss()
dir_checkpoint = Path('/kaggle/working/checkpoints/')
test_score = evaluate('testing round',args, model.cuda(), test_loader, device, args.amp, criterion = criterion,dir_checkpoint =dir_checkpoint)

run.stop()

