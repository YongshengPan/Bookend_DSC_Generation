import os
from multiprocessing import Process
from itertools import combinations
from dataset_dsc import DataBase
from core.multimodels_comb import MultiModels

def main_task(model_stats='extra_test',
              src_images=('T1_C', 'T2_FLAIR', 'ADC', 'T1WI', 'T2WI',),
              tar_images=('CBV', ),
              synthesize_backbone=('simpleunet', 'simpleunet'),
              synthesize_downdeepth=(2, 2),
              src_losses=(['dis', 'p2p', 'msl'], ['p2p', 'dis']),
              training_modules=('synA', 'synB', 'advA', 'advB'),
              restore_models=None,
              input_path="D:/dataset/Bookend_DSC_Generation/"
              ):
    num_of_splits = 4
    num_of_train_splits = 3
    dataset = 'dsc_new'
    version = 'v1'
    output_path = "./outputseg/" + dataset + "/"

    if restore_models is None:
        restore_models = {'clsA': None, 'clsB': None, 'synA': None, 'synB': None, 'advA': None, 'advB': None}

    # model_stats = 'extra_test'
    # src_losses = [['dis', 'p2p', 'msl'], ['p2p', 'dis']]
    # training_modules = ('synA', 'synB', 'advA', 'advB')
    # tar_images = ['CBV', ],
    # src_images = ['T1_C', 'T2_FLAIR', 'ADC', 'T1WI', 'T2WI',]
    tar_images = list(tar_images)
    src_images = list(src_images)
    src_losses = sorted(src_losses[0]), sorted(src_losses[1])
    src_images = sorted(src_images)
    subdir = 'scr_images_{0}_synA_{1}_synB_{2}'.format(''.join(src_images), ''.join(src_losses[0]), ''.join(src_losses[1]))
    image_shape = (256, 288, 48)


    database = DataBase(input_path,
                        side_len=(320, 320, 48),
                        center_shift=(0, 0, 0),
                        data_shape=image_shape,
                        aug_side=(16, 16, 96),
                        aug_stride=(16, 16, 16),
                        model="once",
                        num_of_splits=num_of_splits,
                        num_of_train_splits=num_of_train_splits,
                        train_combinations=None,
                        cycload=True,
                        use_augment=True,
                        useGT=True,
                        submodalities=[src_images, tar_images],
                        randomcrop=(0, 1),
                        randomflip=('sk', 'flr', 'fud', 'r90'))

    clsAB_template = {'network': 'simpleclassifier',
                      'input_shape': image_shape,
                      'activation': 'softmax',
                      'basedim': 16,
                      'input_alter': ['CBV', ],
                      'input_const': [],
                      'output': ['label', ],
                      'use_spatial_kernel': True,
                      'use_second_order': True
                      }
    clsA_params = clsAB_template.copy()
    clsA_params.update({'input_alter': tar_images, })
    clsB_params = clsAB_template.copy()
    clsB_params.update({'input_alter': tar_images, })

    synAB_template = {'network': synthesize_backbone[0],
                      'input_shape': image_shape,
                      'downdeepth': synthesize_downdeepth[0],
                      'task_type': 'synthesis', 'activation': 'relu',
                      'output_channel': 1, 'basedim': 8,
                      'input_alter': [], 'input_const': src_images,
                      'output': tar_images, 'losses': src_losses[0],
                      'use_fake': -0.5
                      }
    synA_params = synAB_template.copy()
    synA_params.update(
        {'network': synthesize_backbone[0], 'downdeepth': synthesize_downdeepth[0], 'input_const': src_images,
         'output': tar_images, 'losses': src_losses[0]})
    synB_params = synAB_template.copy()
    synB_params.update(
        {'network': synthesize_backbone[1], 'downdeepth': synthesize_downdeepth[1], 'input_const': src_images,
         'output': tar_images, 'losses': src_losses[1]})

    model = MultiModels(database, output_path,
                        subdir=subdir,
                        basedim=16,
                        batchsize=1,
                        model_type='2.5D',
                        numchs=database.channels,
                        training_modules=training_modules,
                        network_params={
                            'clsA': clsA_params,
                            'clsB': clsB_params,
                            'synA': synA_params,
                            'synB': synB_params,
                            'advA': {'network': 'generaldiscriminator',
                                     'model_type': 'normal',
                                     'input_shape': image_shape,
                                     'input_alter': ['CBV', ],
                                     'input_const': [],
                                     'output': None,
                                     'activation': None,
                                     'basedim': 16
                                     },
                            'advB': {'network': 'generaldiscriminator',
                                     'model_type': 'normal',
                                     'input_shape': image_shape,
                                     'input_alter': ['CBV', ],
                                     'input_const': [],
                                     'output': None,
                                     'activation': None,
                                     'basedim': 16
                                     }},
                        max_num_images=500,
                        cls_num_epoch=[],
                        syn_num_epoch=list(range(0, 500)),
                        learning_rate=0.001)

    model.model_setup()
    start_epoch = 0
    for item in restore_models:
        if restore_models[item] is not None:
            model.model_load(restore_models[item], [item])
            start_epoch = max(start_epoch, restore_models[item])

    if model_stats == 'train':
        model.train(start_epoch, inter_epoch=10)
    elif model_stats == 'cross_validation':
        model.cross_validation()
    elif model_stats == 'test':
        model.test()
    elif model_stats == 'extra_test':
        model.test_synthesis()
    else:
        print('unknown model_stats:', model_stats)


def squence_task(model_stats, allcombs, devices=0, src_losses=(['dis', 'p2p', 'msl'], ['dis', 'p2p', 'msl', 'ct_dice'])):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(devices)
    for src_images in allcombs:
        # restore_models = {'clsA': None, 'clsB': None, 'synA': 500, 'synB': 500, 'advA': 500, 'advB': 500}
        restore_models = {'clsA': None, 'clsB': None, 'synA': None, 'synB': None, 'advA': None, 'advB': None}
        print(src_images)
        main_task(model_stats=model_stats, src_images=src_images, tar_images=['CBV', ],
                  restore_models=restore_models, synthesize_backbone=('simpleunet', 'simpleunet'),
                  synthesize_downdeepth=(4, 4),
                  src_losses=src_losses)


if __name__ == '__main__':
    nameofGPUs = [0, ]
    numofthread = len(nameofGPUs)
    modalities = ('T1_C', 'T2_FLAIR', 'ADC', 'T1WI', 'T2WI',)
    allcombs = list(combinations(modalities, 1))+list(combinations(modalities, 2))+list(combinations(modalities, 3))\
               +list(combinations(modalities, 4))+list(combinations(modalities, 5))
    allcombs = list(combinations(modalities, 5))

    allcombs = (('T1_C', 'T2_FLAIR', 'ADC', 'T1WI', 'T2WI',), ('T1_C', 'T2_FLAIR', 'T1WI', 'T2WI',))
    processes = [#Process(target=squence_task, args=('extra_test', allcombs[0:1], nameofGPUs[0 % numofthread], (['dis', 'msl', 'p2p'], ['dis', 'p2p', ]))),
                 #Process(target=squence_task, args=('extra_test', allcombs[0:1], nameofGPUs[1 % numofthread], (['p2p', 'msl'], ['p2p', ]))),
                 Process(target=squence_task, args=('train', allcombs[1:2], nameofGPUs[2 % numofthread], (['dis', 'msl', 'p2p'], ['dis', 'p2p', ]))),
                 Process(target=squence_task, args=('train', allcombs[1:2], nameofGPUs[3 % numofthread], (['p2p', 'msl'], ['p2p', ])))]

    for groupID in range(0, len(processes), numofthread):
        param_group = processes[groupID:groupID + numofthread]
        for pro in param_group:
            pro.start()
        for pro in param_group:
            pro.join()


    # for pro in processes:
    #     pro.start()
    # for pro in processes:
    #     pro.join()

    # processes = [Process(target=squence_task, args=('extra_test', allcombs[idx::numofthread], idx%4, [['p2p', 'msl'], ['p2p', ]])) for idx in range(numofthread)]
    # for pro in processes:
    #     pro.start()
    # for pro in processes:
    #     pro.join()


