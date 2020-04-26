class PLBasicImageClassificationSystem(pl.LightningModule):
    
    def __init__(self, model, hparams):
    #def __init__(self, train_loader, val_loader, model):
        super(PLBasicImageClassificationSystem, self).__init__()
        #self.train_loader = train_loader
        #self.val_loader = val_loader
        self.hparams = hparams
        self.model = model
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
# For Training
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        log = {'avg_train_loss': avg_loss}
        return {'avg_train_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'avg_val_loss'}]
    
# For Validation
    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)

        return {'val_loss': val_loss, 'y': y, 'y_hat': y_hat}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        
        y = torch.cat([x['y'] for x in outputs]).cpu().detach().numpy().copy()
        y_hat = torch.cat([x['y_hat'] for x in outputs]).cpu().detach().numpy().copy()

        preds = np.argmax(y_hat, axis=1)
        val_acc = metrics.accuracy_score(y, preds)
        val_qwk = metrics.cohen_kappa_score(y, preds, weights='quadratic')


        log = {'avg_val_loss': avg_loss, 'val_acc': val_acc, 'val_qwk': val_qwk}
        return {'avg_val_loss': avg_loss, 'log': log}

# For Data
    def prepare_data(self):
        train_df = pd.read_csv(os.path.join(self.hparams.data_dir,'train.csv'))
        train_df, val_df = train_test_split(train_df, stratify=train_df['isup_grade'])

        transform = A.Compose([A.Resize(height=self.hparams.image_size, width=self.hparams.image_size, interpolation=1, always_apply=False, p=1.0),
                     A.Flip(always_apply=False, p=0.5),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                      ])
        
        self.train_dataset = PANDADataset(train_df, self.hparams.data_dir, self.hparams.image_format, transform=transform)
        self.val_dataset = PANDADataset(val_df, self.hparams.data_dir, self.hparams.image_format, transform=transform)
        
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=4)

    #def test_dataloader(self):
    #    # OPTIONAL
    #    pass
