#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np

def get_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def distribution_noniid(dataset_labels, num_users, num_classes=10, beta=0.5):
    """
    Sample non-I.I.D client data from provided dataset labels
    :param dataset_labels: data labels with equal sized classes
    :param num_users: number of users
    :param num_classes: number of all available classes
    :param beta: takes value between 0 and 1. Lower beta causes higher imbalance.
    :return dict_users: dictionary with each clients 
    index as key and image indexes list as value
    """
    
    # MNIST: dataset.train_labels or CIFAR: dataset.targets
    labels = np.array(dataset_labels)                 
    data_size = labels.shape[0]  # len(dataset)
    idxs = np.arange(data_size)

    if num_users*num_classes > data_size:
        raise ValueError("Not enough data. Provided data size must be at least num_users*num_classes: {}".format(num_users*num_classes))

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    required_min_items_per_user = 10    
    min_item_user = 0    
                                         
    class_per_user = idxs_labels[0, :].reshape(num_classes, int(data_size/num_classes)) 
    filter = np.ones((num_classes, num_users))
    selected_users = [[] for i in range(num_users)] 

    def maxitem_per_user(users, filters, portions):
      for i, k in enumerate(users):                     
        if len(k) > data_size / num_users :
          filters[:, i] = filters[:, i]*0
      return filters * portions, filters                

    while min_item_user < required_min_items_per_user:   

        np.random.shuffle(np.transpose(class_per_user))

        class_portions_peruser = np.repeat(np.random.dirichlet(np.repeat(beta, num_users)), num_classes).reshape(num_classes, num_users) 
        class_portions_peruser, filter = maxitem_per_user(selected_users, filter, class_portions_peruser)
        ##if filter.all() == np.zeros((num_classes, num_users)).all(): break
        class_portions_peruser = np.divide(class_portions_peruser, np.sum(class_portions_peruser, axis=1).reshape(-1,1))
        class_portions_peruser = (np.cumsum(class_portions_peruser, axis=1) * class_per_user.shape[1]).astype(int)[:, :-1]  

        for i in range(num_classes):
            selected_users = [user_i + user_ix.tolist() for user_i, user_ix in zip(selected_users, np.split(class_per_user[i], class_portions_peruser[i]))]
  
        min_item_user = min([len(user_i) for user_i in selected_users]) 

    dict_users = {k: np.random.permutation(v).tolist() for k, v in enumerate(selected_users)} 

    return dict_users

