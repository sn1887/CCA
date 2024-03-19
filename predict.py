from torch.utils.data import DataLoader

model = VisionTransformer(TRANSCONFIG['R50-ViT-B_16'], img_size=256, num_classes=1, zero_head=False, vis=True)
model = nn.DataParallel(model)
args.load = '/kaggle/working/checkpoints/best_checkpoint.pth'
state_dict = torch.load(args.load, map_location=device)
del state_dict['mask_values']
model.load_state_dict(state_dict)
logging.info(f'Model loaded from {args.load}')


test_set   = MedicalImageDataset('test',noise_typ=None)
loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
test_loader = DataLoader(test_set, shuffle=True, **loader_args)

criterion = nn.CrossEntropyLoss() if args.classes > 1 else DiceBCELoss()
dir_checkpoint = Path('/kaggle/working/checkpoints/')
test_score = evaluate('testing round',args, model.cuda(), test_loader, device, args.amp, criterion = criterion,dir_checkpoint =dir_checkpoint)
