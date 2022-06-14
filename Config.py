class OutpaintingConfig(object):
    
    LEARNING_RATE = 3e-4
    
    ADAM_BETAS = (0.5, 0.999)
    
    BATCH_SIZE = 8
    
    CROPPED_SIZE = 64
    
    OUTPUT_SIZE = 128
    
    LOSS_WEIGHTS = {
        'PIXEL': 0.5,
        'PER': 0.5,
        'ADV': 0.0004
    }
    
    PIXEL_LOSS = 'L1' # 'L1' for L1Loss or 'MSE' for MSELoss
    
    PER_LOSS = 'SSIM' # 'SSIM' or 'VGG'

    #def __init__(self):
        

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")