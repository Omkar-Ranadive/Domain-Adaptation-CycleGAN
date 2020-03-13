"""
Iteratively pass back the generated output back as input
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np


def run_iterative_gan(dataset, model, opt, web_dir, iterations):
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # opt.num_test = 130
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    print("Num test", opt.num_test)
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        # print(data.keys())
        # cur_data = data['A'].to(model.device)
        # print(cur_data.cpu().numpy().shape)
        # cur_data = cur_data.cpu().numpy()[0]
        # axes[0].imshow(np.moveaxis(cur_data, [0, 1, 2], [2, 0, 1]))

        cur_data = data
        print(cur_data['A_paths'])
        # Get the image name to keep things simple while saving
        cur_path = cur_data['A_paths']
        for it in range(iterations):
            print("Iteration: ", it)
            model.set_input(cur_data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            # axes[i, it+1].imshow(visuals)
            cur_data = {'A': visuals['fake'], 'A_paths': cur_path}  # Stick to the format it is expecting

            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        iterative=True, it_count=it)

        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))

    webpage.save()  # save the HTML


def initialize_model(opt):
    '''
    Execute the parse with --dataroot specified in configuration
    The parse also calls the base test parameters and sets it to default values
    Else everything needs to be set manually
    '''

    # opt = TestOptions().parse()
    # Manually set the test values for now
    # opt.dataroot = './datasets/summer2winter_yosemite'
    # opt.name = name
    # opt.model = model_name
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    # opt.dataset_mode = 'single'
    # opt.max_dataset_size = dataset_size
    # opt.direction = 'AtoB'
    print(opt.dataroot)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    print(opt.results_dir)
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory

    return dataset, model, opt, web_dir


if __name__ == '__main__':
    # Initialize the model
    # Iterative test on summer to winter
    # dataset, model, opt, web_dir = initialize_model(name='s2w_iterative', model_name='test', dataset_size=50)
    # Iterative test on data augmentation
    # dataset, model, opt, web_dir = initialize_model(name='s2w_cyclegan_new', model_name='test', dataset_size=50)
    # Iterative test on simulator to real
    # dataset, model, opt, web_dir = initialize_model(name='s2r', model_name='test', dataset_size=float('inf'))
    # run_iterative_gan(dataset, model, opt, web_dir, iterations=1)
    #
    #
    opt = TestOptions().parse()
    dataset, model, opt, web_dir = initialize_model(opt)
    run_iterative_gan(dataset, model, opt, web_dir, iterations=opt.iterations)

























