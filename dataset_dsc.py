import os
import numpy as np
import cv2
from itertools import combinations
import random
import SimpleITK as sitk
from core.dataproc_utils import resize_image_itk, register, get_aug_crops, standard_normalization, histogram_normalization


class DataBase(object):
    def __init__(self,
                 input_path,
                 side_len=(320, 320, 16),
                 center_shift=(0, 0, 0),
                 data_shape=(256, 288, 16),
                 aug_side=(16, 16, 16),
                 aug_stride=(8, 8, 8),
                 model="once",
                 num_of_splits=2,
                 num_of_train_splits=1,
                 train_combinations=None,
                 submodalities=('T1_C', 'T2_FLAIR', 'ADC', 'T1WI', 'T2WI', 'CBV'),
                 cycload=True,
                 use_augment=True,
                 useGT=True,
                 randomcrop=(0, 1),
                 randomflip=('sk', 'flr', 'fud', 'r90')):
        self.side_len = side_len
        self.input_path = input_path
        self.center_shift = center_shift
        self.data_shape = data_shape
        self.use_augment = use_augment
        self.aug_side = aug_side
        self.aug_stride = aug_stride
        self.num_of_splits = num_of_splits
        self.num_of_train_splits = num_of_train_splits
        self.cycload = cycload
        self.use_augment = use_augment
        self.useGT = useGT
        self.randomcrop = randomcrop
        self.randomflip = randomflip
        self.submodalities = submodalities
        self.cls_num = 2
        self.channels = {'T1WI': 1, 'T2WI': 1, 'T2_FLAIR': 1, 'T1_C': 1, 'ADC': 1, 'bm': 1, 'CBV': 1, 'CBF': 1, 'MTT': 1, 'label': 2}
        self.group_map = {'DM': 1, 'AD': 1, 'CN': 0, 'pMCI': 1, 'sMCI': 0, 'sSCD': 0, 'pSCD': 1, 'MCI': 1, 'sSMC': 0,
                          'pSMC': 1, 'SMC': 0, 'sCN': 0, 'pCN': 0, 'ppCN': 1, 'Autism': 1, 'Control': 0}
        if model == 'cross_validation':
            self.train_combinations = list(combinations(range(num_of_splits), num_of_train_splits))
        elif model == 'once':
            if train_combinations is None:
                self.train_combinations = [list(combinations(range(num_of_splits), num_of_train_splits))[0]]
            else:
                self.train_combinations = [train_combinations]
        self.dataset = 'dsc'
        self.datapool = {}
        self.input_setup()

    def input_setup(self):
        # imdb_dsc0 = [[os.path.join('for_figures', subdir, datedir), 0] for subdir in os.listdir(os.path.join(self.input_path, 'for_figures'))
        #              for datedir in os.listdir(os.path.join(self.input_path, 'for_figures', subdir))]

        imdb_dsc1 = [[os.path.join('first_part', subdir, datedir), 0] for subdir in os.listdir(os.path.join(self.input_path, 'first_part'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'first_part', subdir))]
        imdb_dsc2 = [[os.path.join('second_part', subdir, datedir), 1] for subdir in os.listdir(os.path.join(self.input_path, 'second_part'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'second_part', subdir))]
        imdb_dsc3 = [[os.path.join('third_part/need_reslice', subdir, datedir), 0] for subdir in os.listdir(os.path.join(self.input_path, 'third_part/need_reslice'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'third_part/need_reslice', subdir))]
        if False:
            imdb_hos4 = [[os.path.join('Hospital 4/xnykd/Part 1', subdir, datedir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 4/xnykd/Part 1'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'Hospital 4/xnykd/Part 1', subdir))] + \
                    [[os.path.join('Hospital 4/xnykd/Part 2', subdir, datedir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 4/xnykd/Part 2'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'Hospital 4/xnykd/Part 2', subdir))] + \
                    [[os.path.join('Hospital 4/xnykd/Part 3', subdir, datedir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 4/xnykd/Part 3'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'Hospital 4/xnykd/Part 3', subdir))]

            imdb_hos3 = [[os.path.join('Hospital 3/slyy/Part 1', subdir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 3/slyy/Part 1'))] + \
                    [[os.path.join('Hospital 3/slyy/Part 2', subdir, datedir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 3/slyy/Part 2'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'Hospital 3/slyy/Part 2', subdir))]

            imdb_hos2 = [[os.path.join('Hospital 2/qyfy', subdir, datedir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 2/qyfy'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'Hospital 2/qyfy', subdir))]

            imdb_hos1 = [[os.path.join('Hospital 1/gd', subdir, datedir), 0] for subdir in
                     os.listdir(os.path.join(self.input_path, 'Hospital 1/gd'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'Hospital 1/gd', subdir))]

            imdb_tcga1 = [[os.path.join('TCGA', subdir, datedir), 0] for subdir in ['TCGA-LGG', ]
                     #os.listdir(os.path.join(self.input_path, 'TCGA'))
                     for datedir in os.listdir(os.path.join(self.input_path, 'TCGA', subdir))]

        print(imdb_dsc1)
        print(imdb_dsc2)
        self.imdb_train_split = [imdb_dsc3, imdb_dsc1]
        self.imdb_valid_split = [imdb_dsc2, imdb_dsc2]
        self.imdb_train = self.imdb_train_split[0]
        self.imdb_valid = self.imdb_valid_split[0]
        self.imdb_test = imdb_dsc1
        print(len(self.imdb_train))
        print(len(self.imdb_test))

    def read_images(self, flnm):
        source_images = {'T1WI': 'T1WI', 'T2WI': 'T2WI', 'T2_FLAIR': 'T2WI_FLAIR', 'T1_C': 'T1WI_C', 'bm': 'bmT1WI', 'ADC': 'ADC'}
        target_images = {'CBV': 'CBV', 'CBF': 'CBF', 'MTT': 'MTT'}
        source_images = {'T1WI': 'T1WI', 'T2WI': 'T2WI', 'T2_FLAIR': ['T2_FLAIR', 'T2WI_FLAIR'], 'T1_C': ['T1WI_C', 'T1WI+C', 'T1_C'], 'bm': 'bmT1WI', 'ADC': 'ADC'}
        target_images = {'CBV': ['CBV', 'bmT1WI'], 'CBF': 'bmT1WI', 'MTT': 'bmT1WI'}
        dataessamble = dict(source_images, ** target_images)

        spacing = [0.6875, 0.6875, 3.0]
        # spacing = [0.3594, 0.3594, None]
        newsize = [320, 320, 50]
        bmpath = os.path.join(self.input_path, flnm, 'bmT1WI.nii.gz')
        T1path = os.path.join(self.input_path, flnm, 'T1WI.nii.gz')
        brainmask = resize_image_itk(sitk.ReadImage(bmpath), newSpacing=spacing, newSize=newsize)
        T1image = resize_image_itk(sitk.ReadImage(T1path), newSpacing=spacing, newSize=newsize)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(brainmask)
        # T1image = resampler.Execute(sitk.ReadImage(T1path))
        # T1image = T1image * sitk.Cast(brainmask / 128, T1image.GetPixelID())
        fix_spacing, fix_size, fix_origin = brainmask.GetSpacing(), brainmask.GetSize(), brainmask.GetOrigin()
        affine = {'spacing': brainmask.GetSpacing(), 'origin': brainmask.GetOrigin(),
                  'direction': brainmask.GetDirection(), 'size': brainmask.GetSize(),
                  'depth': brainmask.GetDepth(), 'dimension': brainmask.GetDimension()}
        viewsample = []
        for ids in dataessamble:
            if ids in self.submodalities[0] + self.submodalities[1]:
                if isinstance(dataessamble[ids], list):
                    for moda in dataessamble[ids]:
                        fullpath = os.path.join(self.input_path, flnm, moda + '.nii.gz')
                        fullpathr = os.path.join(self.input_path, flnm, moda + 'r.nii.gz')
                        if os.path.exists(fullpath):
                            break
                else:
                    fullpath = os.path.join(self.input_path, flnm, dataessamble[ids] + '.nii.gz')
                    fullpathr = os.path.join(self.input_path, flnm, dataessamble[ids] + 'r.nii.gz')
                if os.path.exists(fullpath):
                    if False and os.path.exists(fullpathr):
                        std_image = sitk.ReadImage(fullpathr)
                    else:
                        # if os.path.exists(fullpath.replace('.nii.gz', '.nrrd')):
                        #     input_image = sitk.ReadImage(fullpath.replace('.nii.gz', '.nrrd'))
                        #     sitk.WriteImage(input_image, fullpath)
                        input_image = resize_image_itk(sitk.ReadImage(fullpath), newSpacing=spacing, newSize=newsize)
                        # stasts = sitk.StatisticsImageFilter()
                        # stasts.Execute(input_image)
                        # input_image = input_image/stasts.GetMaximum()*5000
                        # input_image = sitk.ReadImage(fullpath)
                        #
                        try:
                            tx2 = register(T1image, input_image)
                            resampler.SetTransform(tx2)
                        except:
                            print(fullpath)
                            tx1 = sitk.CenteredTransformInitializer(sitk.Cast(brainmask, input_image.GetPixelID()),
                                                                sitk.Cast(input_image, input_image.GetPixelID()),
                                                                sitk.Euler3DTransform(),
                                                                operationMode=sitk.CenteredTransformInitializerFilter.GEOMETRY)
                            resampler.SetTransform(tx1)
                        resampler.SetReferenceImage(brainmask)

                        rsz_image = resampler.Execute(input_image)
                        rsz_image = sitk.Cast(rsz_image, sitk.sitkInt16) * sitk.Cast(brainmask / 128, sitk.sitkInt16)
                        std_image = standard_normalization(rsz_image, remove_tail=False, divide='mean')*1000
                        std_image = sitk.Cast(std_image, sitk.sitkInt16)
                        sitk.WriteImage(std_image, fullpathr)

                    input_data = np.transpose(np.float32(sitk.GetArrayFromImage(std_image))/1000.0)
                    dataessamble[ids] = np.expand_dims(input_data, -1)
                    # dataessamble[ids] = np.minimum(np.expand_dims(input_data, -1) * 2 - 1, 1)
                    viewsample.append(input_data[:, :, np.shape(input_data)[2]//2])
                else:
                    dataessamble[ids] = None
       #print(np.mean(dataessamble['CBV'][dataessamble['CBV'] > 0]), np.std(dataessamble['CBV'][dataessamble['CBV'] > 0]))
        dataessamble['affine'] = affine
        viewpath = os.path.join(self.input_path, flnm.replace('\\', '_').replace('/', '_') + '.png')
        cv2.imwrite(viewpath, np.concatenate(viewsample, axis=0)*255)
        return dataessamble

    def inputAB(self, imdb=None, split=0, index=0, model='train', aug_model='random', aug_count=1, aug_index=(1,)):
        if imdb is None:
            imdb = self.imdb_train_split[split] if model == 'train' else self.imdb_valid_split[split]
        flnm, group = imdb[index][0:2]
        # print(flnm)
        if flnm in self.datapool:
            dataessamble = self.datapool[flnm]
        else:
            label = np.zeros(2, np.float32)
            label[group] = 1
            dataessamble = self.read_images(flnm)
            dataessamble.update({'label': label})
            if self.cycload:
                self.datapool[flnm] = dataessamble
        refdata = dataessamble[self.submodalities[0][0]]

        aug_side = self.aug_side
        aug_step = np.maximum(self.aug_stride, 1)
        image_size = np.shape(refdata)[0], np.shape(refdata)[1], np.shape(refdata)[2]

        aug_range = [min(aug_side[dim], (image_size[dim] - self.data_shape[dim] - self.center_shift[dim]) // 2) for dim in range(3)]
        aug_center = [(image_size[dim] + self.center_shift[dim] - self.data_shape[dim]) // 2 for dim in range(3)]
        if not self.use_augment: aug_model = 'center'
        aug_crops, count_of_augs = get_aug_crops(aug_center, aug_range, aug_step,
                                                 aug_count=aug_count, aug_index=aug_index, aug_model=aug_model)
        aug_crops = [[sX1, sY1, sZ1, sX1 + self.data_shape[0], sY1 + self.data_shape[1], sZ1 + self.data_shape[2]] for
                     sX1, sY1, sZ1 in aug_crops]

        datainput = {'orig_size': dataessamble['affine']['size'], 'aug_crops': aug_crops, 'count_of_augs': [count_of_augs], 'affine': [dataessamble['affine']]}
        for it in dataessamble:
            if it in self.submodalities[0] + self.submodalities[1]:
                datainput[it] = np.concatenate([
                    dataessamble[it][np.newaxis, sX1:sX2, sY1:sY2, sZ1:sZ2] for sX1, sY1, sZ1, sX2, sY2, sZ2 in aug_crops], axis=0)
        datainput['label'] = dataessamble['label'][np.newaxis, :]
        return flnm, datainput

    def expand_apply_synthesis(self, inputA, syn_func):
        sinputA = inputA
        if np.ndim(sinputA) == 4:
            sinputA = sinputA[np.newaxis]
        elif np.ndim(sinputA) == 3:
            sinputA = sinputA[np.newaxis, :, :, np.newaxis]
        sfake_B = syn_func(sinputA)
        fake_B = sfake_B[0]
        return fake_B

    def evaluate_output(self, imdb=None, split=0, index=0, model='test', evalfunc=None, aug_side=(8, 8, 4), crop_output=False):
        if imdb is None:
            imdb = self.imdb_train_split[split] if model == 'train' else self.imdb_valid_split[split]
        flnm, group = imdb[index][0:2]
        label = np.zeros(2, np.float32)
        label[group] = 1
        dataessamble = self.read_images(flnm)
        refdata, affine = dataessamble[self.submodalities[0][0]], dataessamble['affine']
        shp = np.shape(refdata)
        syn_A = np.zeros((shp[0], shp[1], shp[2], 1), np.float32)
        syn_B = np.zeros((shp[0], shp[1], shp[2], 1), np.float32)
        cusum = np.zeros((shp[0], shp[1], shp[2], 1), np.float32)
        cX, cY, cZ = shp[0] // 2, shp[1] // 2, shp[2] // 2
        aX = min(aug_side[0], cX - self.side_len[0] // 2)
        aY = min(aug_side[1], cY - self.side_len[1] // 2)
        aZ = min(aug_side[2], cZ - self.side_len[2] // 2)
        src_img = np.concatenate([dataessamble[it] for it in self.submodalities[0]], axis=-1)
        tar_img = np.concatenate([dataessamble[it] for it in self.submodalities[1]], axis=-1)
        for fctr in range(50):
            idx = random.randint(-aX, aX)
            idy = random.randint(-aY, aY)
            idz = random.randint(-aZ, aZ)
            sX = cX - self.side_len[0] // 2 + idx
            sY = cY - self.side_len[1] // 2 + idy
            sZ = cZ - self.side_len[2] // 2 + idz
            im_all = src_img[np.newaxis, sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ]
            sf_c = evalfunc['synA'](im_all)
            sf_s = evalfunc['synB'](im_all)

            syn_A[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ] += sf_c[0]
            syn_B[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ] += sf_s[0]
            cusum[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ] += 1
        syn_A = syn_A / (cusum + 1.0e-6)
        syn_B = syn_B / (cusum + 1.0e-6)
        # print(syn_A.dtype)
        synA = np.where(cusum > 0.5, syn_A, np.min(syn_A))
        synB = np.where(cusum > 0.5, syn_B, np.min(syn_B))
        sX = cX - self.side_len[0] // 2
        sY = cY - self.side_len[1] // 2
        sZ = cZ - self.side_len[2] // 2
        if crop_output:
            src_img = src_img[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ]
            tar_img = tar_img[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ]
            synA = synA[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ]
            synB = synB[sX:self.side_len[0] + sX, sY:self.side_len[1] + sY, sZ:self.side_len[2] + sZ]
        eval_out = {'flnm': flnm, 'refA': tar_img, 'refB': tar_img, 'synB': synB, 'synA': synA, 'pred': label,
                    'label': label, 'affine': affine}
        return eval_out

    def save_output(self, result_path, flnm, eval_out):
        flnm, refA, refB, synA, synB = eval_out['flnm'], eval_out['refA'], eval_out['refB'], eval_out['synA'], eval_out['synB']
        affine = eval_out['affine']
        if isinstance(flnm, bytes): flnm = flnm.decode()
        result_path = result_path.replace('dsc_new', 'for_figures')
        if not os.path.exists(result_path + "/{0}".format(flnm)): os.makedirs(result_path + "/{0}".format(flnm))
        synA = (synA + 0.0) * 128.0
        synB = (synB + 0.0) * 128.0
        refA = (refA + 0.0) * 128.0
        refB = (refB + 0.0) * 128.0
        for ref in ['refA', 'refB', 'synA', 'synB']:
            img = eval_out[ref]
            if img is not None:
                img = sitk.GetImageFromArray(np.round((np.transpose(img, axes=(2, 1, 0, 3))) * 2000))
                # img = sitk.Cast(img, sitk.sitkInt16)
                # img = sitk.GetImageFromArray(self.ct_rgb2gray(img))
                img.SetOrigin(affine['origin'])
                img.SetSpacing(affine['spacing'])
                img.SetDirection(affine['direction'])
                sitk.WriteImage(img, result_path + "/{0}/{1}.nii.gz".format(flnm, ref), useCompression=True)
        # if synA is not None:
        #     nib.save(nib.Nifti1Image(np.array(synA, np.float32), affine=affine), result_path + "/{0}/CBV_SE1.nii.gz".format(flnm))
        #     # print(result_path)
        # if refA is not None:
        #     nib.save(nib.Nifti1Image(np.array(refA, np.float32), affine=affine), result_path + "/{0}/CBV_GT1.nii.gz".format(flnm))
        # if synB is not None:
        #     nib.save(nib.Nifti1Image(np.array(synB, np.float32), affine=affine), result_path + "/{0}/CBV_SE2.nii.gz".format(flnm))
        # if refB is not None:
        #     nib.save(nib.Nifti1Image(np.array(refB, np.float32), affine=affine), result_path + "/{0}/CBV_GT2.nii.gz".format(flnm))
        if eval_out['synA'] is not None and eval_out['synB'] is not None and eval_out['refA'] is not None:
            cv2.imwrite(result_path + "/{0}/CBV_sample.png".format(flnm),
                        np.concatenate((refA[:, :, 13], synA[:, :, 13], synB[:, :, 13]), axis=0))

