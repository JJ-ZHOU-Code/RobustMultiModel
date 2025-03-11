from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

# from utils.utils import generate_split, nth
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd 


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self, csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False, 
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)


        slide_data = pd.read_csv(csv_path, low_memory=False)
        ### new
        missing_slides_ls = ['TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs',
                             'TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs',
                             'TCGA-HT-7483-01Z-00-DX1.7241DF0C-1881-4366-8DD9-11BF8BDD6FBF.svs',
                             'TCGA-06-0882-01Z-00-DX2.7ad706e3-002e-4e29-88a9-18953ba422bf.svs']
        slide_data.drop(slide_data[slide_data['slide_id'].isin(missing_slides_ls)].index, inplace=True)
        missing_slides_csv = '/home/zjj/zjj/data/TCGA/gbmlgg/ExpData/tiles-l1-s256/missing_h5.csv'
        missing_slides_df = pd.read_csv(missing_slides_csv)
        missing_slides_ls = missing_slides_df['slide_id'].to_list()
        slide_data.drop(slide_data[slide_data['slide_id'].isin(missing_slides_ls)].index, inplace=True)

        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy() # 移除 slide_data 数据框中在指定列 case_id 上的重复行，并保留每个 case_id 的第一个出现的行
        uncensored_df = patients_df[patients_df['censorship'] < 1] # 删失状态 0为事件发生，1为失访

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False) # 根据数据的分位数进行分箱，使每个箱子中的数据量大致相等。
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True) # 根据指定的边界进行分箱，可以用于定义自定义的分箱边界。
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}  # 字典信息：patient_id(key), 该病人对应的所有slide_id(value)
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id']) # 将slide_id替换为case_id

        label_dict = {} # 存储一个label_dict字典，其中key为元组(q_bins的四个label, 0/1) 其中0/1对应是否失访。所以一个有8个key，value则是每个key分别对应0-7
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict # 根据label_dict字典，给df重新设置disc_label和label。其中disc_label为q_bins划分出的4个label，label则是刚刚的8个key对应的8个value标签
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data # 经过前两行，调整了列的顺序，把后两列移到了最前面。
        self.metadata = slide_data.columns[:12] # 前12列为metadata
        self.mode = mode
        self.cls_ids_prep()

        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, 
                                  signatures=self.signatures, data_dir=self.data_dir, 
                                  label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split

    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split#, test_split

    def create_splits(self, which_splits, split_dir, random_state=7):
        kf = KFold(n_splits=5, random_state=random_state, shuffle=True)
        df_smiles = self.slide_data['slide_id'].drop_duplicates()
        print(df_smiles.shape)
        # print(df_smiles)
        for i, (train_index, test_index) in enumerate(kf.split(df_smiles)):
            # 获取训练集和测试集的smiles
            train_smiles = df_smiles.iloc[train_index].reset_index(drop=True)
            test_smiles = df_smiles.iloc[test_index].reset_index(drop=True)
            # print(train_smiles)
            # combine_set = pd.concat([train_set['smiles'], test_set['smiles']], ignore_index=True, axis=1)
            combine_set = pd.concat([train_smiles, test_smiles], ignore_index=True, axis=1)
            combine_set.columns = ['train', 'val']
            # 保存训练集和测试集到csv文件
            save_fold_dir = os.path.join('splits', which_splits, split_dir)
            os.makedirs(save_fold_dir, exist_ok=True)
            combine_set.to_csv(os.path.join(save_fold_dir, f'splits_{i}.csv'), index=True)

            # # 根据smiles在df_cleaned中选择行
            train_set = self.slide_data[self.slide_data['slide_id'].isin(train_smiles)].reset_index(drop=True)
            test_set = self.slide_data[self.slide_data['slide_id'].isin(test_smiles)].reset_index(drop=True)

            print(f'第{i+1}折划分完成, slide_id: ')
            print(train_smiles.shape, test_smiles.shape)
            print(self.slide_data.shape)
            print(self.slide_data[self.slide_data['slide_id'].isin(train_set['slide_id'])].shape)
            print(self.slide_data[self.slide_data['slide_id'].isin(test_set['slide_id'])].shape) 


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)
                elif self.mode == 'pibd':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    patch_features = torch.cat(path_features, dim=0)

                    max_patches = 4096
                    n_samples = min(patch_features.shape[0], max_patches)
                    patch_idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
                    patch_features = patch_features[patch_idx, :]

                    # make a mask
                    if n_samples == max_patches:
                        # sampled the max num patches, so keep all of them
                        mask = torch.zeros([max_patches])
                    else:
                        # sampled fewer than max, so zero pad and add mask
                        original = patch_features.shape[0]
                        how_many_to_add = max_patches - original
                        zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                        patch_features = torch.concat([patch_features, zeros], dim=0)
                        mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])

                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx].values)
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx].values)
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx].values)
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx].values)
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx].values)
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx].values)
                    return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)
                
                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx].values)
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx].values)
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx].values)
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx].values)
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx].values)
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx].values)
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures


        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))
        
        
        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        # print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--