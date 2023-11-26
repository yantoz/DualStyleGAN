dependencies = [
    'torch', 'numpy',
    'cv2', 'dlib', 'torchvision',
]

import os
import cv2
import torch
import bz2
import dlib
import tempfile
import numpy as np
import torchvision

from torchvision import transforms
import torch.nn.functional as F

from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from model.encoder.align_all_parallel import align_face

from argparse import Namespace

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("dualstylegan")

class _DualStyleGAN():
    
    def __init__(self, generator, exstyle, encoder, predictor,
        style_degree=0.5, color_transfer=True, device='cpu'):
        self._generator = generator
        self._exstyle = exstyle
        self._encoder = encoder
        self._predictor = predictor
        self.style_degree = style_degree
        self.color_transfer = color_transfer
        self.device = device

    def __call__(self, image):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
        with torch.no_grad():

            wplus = False

            z_plus_latent = not wplus
            return_z_plus_latent = not wplus
            input_is_latent = wplus   

            I = align_face(image, self._predictor)
            I = transform(I).unsqueeze(dim=0).to(self.device)

            # reconstructed content image and its intrinsic style code
            img_rec, instyle = self._encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
            img_rec = torch.clamp(img_rec.detach(), -1, 1)

            latent = self._exstyle
            if not self.color_transfer:
                latent[:,7:18] = instyle[:,7:18]
            # extrinsic styte code
            exstyle = self._generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
           
            weights = [self.style_degree]*4 + [self.style_degree]*3 + [int(self.color_transfer)]*11
            log.debug("Weights: {}".format(weights))

            # style transfer 
            # input_is_latent: instyle is not in W space
            # z_plus_latent: instyle is in Z+ space
            # use_res: use extrinsic style path, or the style is not transferred
            # interp_weights: weight vector for style combination of two paths

            img_gen, _ = self._generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                truncation=0.75, truncation_latent=0, use_res=True, interp_weights=weights)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)

        output = ((img_gen[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        return (image, output)

def dualstylegan(
    progress=True, map_location=None,
    style="cartoon", style_id=26, style_degree=0.5,
    color_transfer=True
):

    BASE_URL="https://huggingface.co/public-data/DualStyleGAN/resolve/main/models"

    ckpt = "generator.pt"
    checkpoint_url="{}/{}/{}".format(BASE_URL, style, ckpt)

    MODEL_DIR=os.path.join(torch.hub.get_dir(), "checkpoints", "dualstylegan_{}".format(style))
    DUALSTYLEGAN_DIR=os.path.join(torch.hub.get_dir(), "checkpoints", "dualstylegan")

    log.debug("Downloading {}".format(checkpoint_url))
    ckpt = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=MODEL_DIR, map_location=map_location, progress=progress)
    dualstylegan = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    dualstylegan.eval()
    dualstylegan.load_state_dict(ckpt['g_ema'])
    dualstylegan.to(map_location)

    model = "encoder.pt"
    model_url = "{}/{}".format(BASE_URL, model)
    model_path = os.path.join(DUALSTYLEGAN_DIR, model)

    log.debug("Downloading {}".format(model_url))
    ckpt = torch.hub.load_state_dict_from_url(model_url, model_dir=DUALSTYLEGAN_DIR, map_location=map_location, progress=progress)
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'output_size' not in opts:
        opts['output_size'] = 1024    
    opts = Namespace(**opts)
    opts.device = map_location
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(map_location)

    exstyle_url="{}/{}/exstyle_code.npy".format(BASE_URL, style)

    log.debug("Downloading {}".format(exstyle_url))
    os.makedirs(MODEL_DIR, exist_ok=True)
    DST = os.path.join(MODEL_DIR, "exstyle_code.npy")
    if not os.path.isfile(DST):
        torch.hub.download_url_to_file(exstyle_url, DST, hash_prefix=None, progress=progress)
    exstyles = np.load(DST, allow_pickle='TRUE').item()
    stylename = list(exstyles.keys())[style_id]
    exstyle = torch.tensor(exstyles[stylename]).to(map_location)

    MODEL = "shape_predictor_68_face_landmarks.dat"
    DST = os.path.join(DUALSTYLEGAN_DIR, MODEL)
    if not os.path.isfile(DST):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "{}.bz2".format(MODEL))
            url = "http://dlib.net/files/{}.bz2".format(MODEL)
            print("Downloading {} to {}".format(url, path)) 
            torch.hub.download_url_to_file(url, path, progress=progress)
            with bz2.BZ2File(path, 'r') as zip:
                data = zip.read()
                open(DST, 'wb').write(data)
    predictor = dlib.shape_predictor(DST)

    model = _DualStyleGAN(dualstylegan, exstyle, encoder, predictor,
        style_degree=style_degree, color_transfer=color_transfer, device=map_location)
    return model

