import os
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import cv2
from PIL import Image
import time

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from models.psp import pSp


class Inference:
    def __init__(self):
        test_opts = TestOptions().parse()
        if test_opts.resize_factors is not None:
            assert len(
                test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        # update test options with options used during training
        ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
        self.opts = ckpt['opts']
        self.opts.update(vars(test_opts))
        if 'learn_in_w' not in self.opts:
            self.opts['learn_in_w'] = False
        if 'output_size' not in self.opts:
            self.opts['output_size'] = 1024
        self.opts = Namespace(**self.opts)
        self.net = pSp(self.opts)
        self.net.eval()
        self.net.cuda()
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        self.transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

    def run(self, frame):
        tic = time.time()
        img = self.transforms_dict['transform_inference'](Image.fromarray(frame))
        img = torch.reshape(img, (1, 3, 256, 256))
        with torch.no_grad():
            input_cuda = img.cuda().float()
            result_batch = run_on_batch(input_cuda, self.net, self.opts)
            result = tensor2cvimg(result_batch[0])
        tac = time.time()
        print('InferenceTime: {}'.format(tac-tic))
        return result


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


def tensor2cvimg(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy().copy()
    var = cv2.cvtColor(var, cv2.COLOR_RGB2BGR)
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')


# if __name__ == '__main__':
#     run()