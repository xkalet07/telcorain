#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: cnn_utility.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0t
Description:
    Function set for training and using CNN module
    for rain event classification using data from CML.

License:
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr

# Import external packages


# Import local modules
import telcorain.procedures.wet_dry.cnn_custom_model.cnn_telcorain_v22 as cnn

# import telcosense_classification.module.cnn_polz as cnn_orig

""" Variable definitions """


""" Function definitions """

# DONE: make 1 CNN architecture with choice of one output value for whole sample
#           or output size = sample size. input parameter
# TODO: patch sample size output testloss is NaN
# TODO: change the WD ref douwnsampling method
# TODO: Early stopping


def attach_cnn_output_to_xarray(
    ds: xr.Dataset, cnn_output: np.ndarray, sample_size: int, threshold=0.5
) -> xr.Dataset:
    """
    Attach CNN output to xarray Dataset by repeating each prediction over its corresponding segment.
    Each segment of length `sample_size` gets a constant label (0 or 1).

    Parameters:
    ds : xr.Dataset
        Input dataset with a time dimension.
    cnn_output : np.ndarray
        Array of CNN probabilities (length = n_segments).
    sample_size : int
        Number of time steps per CNN input segment.
    threshold : float
        Decision threshold to classify as wet or dry.

    Returns:
    ds : xr.Dataset
        Dataset with a new variable 'wet'.
    """

    n_segments = len(cnn_output)
    total_points = n_segments * sample_size
    wet_array = np.zeros(total_points)

    for i, val in enumerate(cnn_output):
        wet_array[i * sample_size : (i + 1) * sample_size] = int(val >= threshold)

    # Pad to match the dataset length
    wet_array_full = np.full(len(ds.time), np.nan)
    max_fill = min(len(wet_array), len(wet_array_full))
    wet_array_full[:max_fill] = wet_array[:max_fill]

    # Create DataArray and assign
    wet_da = xr.DataArray(
        wet_array_full, coords={"time": ds.time}, dims=["time"], name="wet"
    )

    ds["wet"] = wet_da
    return ds


def cnn_infer_only(
    preprocessed_df: pd.DataFrame,
    param_dir: str,
    num_channels: int = 2,
    sample_size: int = 60,
    batchsize: int = 256,
    kernel_size: int = 3,
    n_conv_filters: list = [16, 32, 64, 128],
    n_fc_neurons: int = 64,
    single_output: bool = True,
):
    """
    Run inference using pretrained CNN on already preprocessed CML data.

    Parameters:
    preprocessed_df : pd.DataFrame
        Output of cml_preprocess(), must contain 'trsl_A', 'trsl_B', etc.
    param_dir : str
        Path to the saved model (relative to cnn_custom_model/ directory).
    num_channels : int
        Number of input features used (e.g., 2 for trsl_A and trsl_B).
    sample_size : int
        Number of time steps in each input segment.
    batchsize : int
        Inference batch size.
    kernel_size : int
        CNN kernel size.
    n_conv_filters : list
        List of filter sizes for each convolutional layer.
    n_fc_neurons : int
        Number of neurons in fully connected layer.
    single_output : bool
        Whether the CNN outputs a single value per segment.

    Returns:
    cnn_out : np.ndarray
        1D array of rain probabilities for each segment.
    """

    # print(preprocessed_df)
    # preprocessed_df.to_csv("preprocessed_df.csv")

    # Use only the expected number of features
    all_possible_inputs = ["trsl_A", "trsl_B", "temperature_rx", "temperature_tx"]
    used_features = all_possible_inputs[:num_channels]

    # Extract and truncate the input matrix
    X = preprocessed_df[used_features].values
    cutoff = len(X) % sample_size
    if cutoff != 0:
        X = X[:-cutoff]

   # --------------------------------------------------------------------------------------------------------------------------
    # TODO: Warning, real reshaping order is [n_samples, n_channels, samplesize]
    #by using .transpose(0,2,1)
    # old: Reshape into (n_samples, sample_size, num_channels)
    X = X.reshape(-1, sample_size, num_channels).transpose(0,2,1)
    # --------------------------------------------------------------------------------------------------------------------------

    # Convert to torch tensors
    X_tensor = torch.Tensor(X)

    # Dummy targets (not used, just for DataLoader compatibility)
    dummy_targets = torch.zeros((X_tensor.shape[0], 1))
    dataset = TensorDataset(X_tensor, dummy_targets)

    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    # Load the CNN model
    model = cnn.cnn_class(
        channels=num_channels,
        sample_size=sample_size,
        kernel_size=kernel_size,
        n_fc_neurons=n_fc_neurons,
        n_filters=n_conv_filters,
        single_output=single_output,
    )
    model.load_state_dict(
        torch.load("telcorain/procedures/wet_dry/cnn_custom_model/" + param_dir)
    )
    model.eval()

    # Run inference
    cnn_output = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.permute(0, 2, 1)
            pred = model(inputs)
            pred = nn.Flatten(0, 1)(pred)
            cnn_output.extend(pred.tolist())

    return np.array(cnn_output).reshape(-1)


def cnn_train(
    ds: pd.DataFrame,
    num_channels=2,
    sample_size=60,
    batchsize=256,
    epochs=20,
    resume_epoch=0,
    learning_rate=0.01,
    kernel_size=3,
    n_conv_filters=[16, 32, 64, 128],
    n_fc_neurons=64,
    single_output=True,
    shuffle=False,
    save_param=False,
):
    """
    Train given cnn modul on given cml dataset over given number of epochs.
    perform testing for each epoch, save training parameters.
    Classification output is one WD probability value for whole sample

    Parameters
    ds : pandas.DataFrame containing CML data and reference rain data for training and testing
    num_channels : int, default = 2, number of variables for cnn to classify from (2*trsl, 2*temperature)
    samplesize : int, default = 100, number of values to be grouped in a sample
    batchsize : int, default = 20, number of samples per batch, to be feed into cnn at once
    epochs : int, default = 20, number of training epochs
    resume_epoch : int, default = 0, if training was performed previouslyover xy epochs,
        continue training at epoch xy+1
    learning_rate : float, default = 0.01, cnn's optimizer learning rate
    kernel_size : int, default = 3
    n_conv_filters : int, number of convolutional layer inputs and outputs
    n_fc_neurons : int, number of neurons in FC layer
    single_output : bool, classification output of the CNN is single value if True, otherwise matches sample_size
    shuffle : bool, default = False, enable torch dataLoader to perform shuffle between epochs if True.
    save_param: boolean, default = False, save training parameters after training

    Returns
    cnn_output: np.array containing 0-1 float classification probability output of CNN
    train_loss: trtainloss during last epoch
    valid_loss: testloss during last epoch
    """

    trsl, ref = get_trsl_and_ref(ds, sample_size, num_channels, single_output)

    k_train = 0.8  # fraction of training data
    train_size = int(len(trsl) * k_train / batchsize) * batchsize

    # Storing as tensors
    train_data = torch.Tensor(trsl[:train_size])
    valid_data = torch.Tensor(trsl[train_size:])

    train_ref = torch.Tensor(ref[:train_size])
    valid_ref = torch.Tensor(ref[train_size:])

    # Turning into TensorDataset
    dataset = torch.utils.data.TensorDataset(train_data, train_ref)
    validset = torch.utils.data.TensorDataset(valid_data, valid_ref)

    trainloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle)
    validloader = torch.utils.data.DataLoader(validset, batchsize, shuffle)

    # CNN model
    model = cnn.cnn_class(
        channels=num_channels,
        sample_size=sample_size,
        kernel_size=kernel_size,
        n_fc_neurons=n_fc_neurons,
        n_filters=n_conv_filters,
        single_output=single_output,
    )

    # used optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )  # dropout alternative , +1 % TP, lower testloss and FP
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )  # each 10 epoch multiply lr by 0.5

    # if resuming training
    if resume_epoch == 0:
        loss_dict = {}
        loss_dict["train"] = {}
        loss_dict["valid"] = {}
        for key in ["train", "valid"]:
            loss_dict[key]["loss"] = []

    # training loop
    cnn_prediction = []
    for epoch in range(resume_epoch, epochs):
        # training
        model.train()
        train_losses = []
        for inputs, targets in tqdm(trainloader):  # meta,
            optimizer.zero_grad()
            pred = model(inputs)  # ,meta)
            # flatten prediction only for single value output
            if single_output:
                pred = nn.Flatten(0, 1)(pred)
            # getting the output
            if epoch == epochs - 1:
                cnn_prediction = cnn_prediction + pred.tolist()

            # calculating the loss function
            loss = nn.BCELoss()(pred, targets)  # BCE = binary cross entropy
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().numpy())
        scheduler.step(loss)
        loss_dict["train"]["loss"].append(np.mean(train_losses))

        # testing
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(validloader):  # meta,
                pred = model(inputs)  # ,meta)
                # flatten prediction only for single value output
                if single_output:
                    pred = nn.Flatten(0, 1)(pred)
                # getting the output
                if epoch == epochs - 1:
                    cnn_prediction = cnn_prediction + pred.tolist()

                loss = nn.BCELoss()(pred, targets)
                valid_losses.append(loss.detach().numpy())
            loss_dict["valid"]["loss"].append(np.mean(valid_losses))

        # learning curve
        print(epoch)
        print("train loss:", np.mean(train_losses))
        print("validation loss:", np.mean(valid_losses))
        print("min validation loss:", np.min(loss_dict["valid"]["loss"]))

        plt.ion()
        if epoch == 0:
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        axs.cla()
        for key in loss_dict.keys():
            for k, key2 in enumerate(loss_dict[key].keys()):
                axs.plot(loss_dict[key][key2], label=key)
                axs.set_title(key2)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        plt.pause(0.0001)
        fig.tight_layout(pad=1.0)
        resume_epoch = epoch
    # export training curve plot
    # fig.savefig('results/loss_curve_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png')
    # plt.close()
    # save cnn parameters
    if save_param:
        path = "telcosense_classification/module/trained_cnn_param/"
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H;%M")
        torch.save(model.state_dict(), (path + date))

    cnn_out = np.array(cnn_prediction).reshape(-1)
    return cnn_out, np.mean(train_losses), np.mean(valid_losses)


def get_trsl_and_ref(
    ds: pd.DataFrame, sample_size=60, num_channels=2, single_output=True
):
    """
    Extract and reshape trsl and ref WD and store them as arrays.
    Cut off samples non fitting into sample size, also reshape ref WD based on single/continuous output

    Parameters
    ds : pandas.DataFrame containing CML data and reference rain data for training and testing
    num_channels : int, default = 2, number of variables for cnn to classify from (2*trsl, 2*temperature)
    sample_size : int, default = 100, number of values to be grouped in a sample
    single_output : bool, classification output of the CNN is single value if True, otherwise matches sample_size

    Returns
    trsl: np.array containing float trsl values in shape [n_samples, n_channels, samplesize]
    ref: np.array containing boolean WD reference in shape [n_samples] or [n_samples, samplesize]
    """
    n_samples = len(ds) // sample_size
    cutoff = len(ds) % sample_size

    if cutoff == 0:
        if num_channels == 2:
            trsl = np.concatenate((ds.trsl_A.values[:], ds.trsl_B.values[:])).reshape(
                (-1, num_channels), order="F"
            )
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0, 2, 1)
        elif num_channels == 4:
            trsl = np.concatenate(
                (
                    ds.trsl_A.values[:],
                    ds.trsl_B.values[:],
                    ds.temp_A.values[:],
                    ds.temp_B.values[:],
                )
            ).reshape((-1, num_channels), order="F")
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0, 2, 1)
        if single_output:
            ref = ds.ref_wd.values[::sample_size]
        else:
            ref = ds.ref_wd.values.reshape((n_samples, sample_size))

    else:
        if num_channels == 2:
            trsl = np.concatenate(
                (ds.trsl_A.values[:-cutoff], ds.trsl_B.values[:-cutoff])
            ).reshape((-1, num_channels), order="F")
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0, 2, 1)
        elif num_channels == 4:
            trsl = np.concatenate(
                (
                    ds.trsl_A.values[:-cutoff],
                    ds.trsl_B.values[:-cutoff],
                    ds.temp_A.values[:-cutoff],
                    ds.temp_B.values[:-cutoff],
                )
            ).reshape((-1, num_channels), order="F")
            trsl = trsl.reshape(n_samples, sample_size, num_channels).transpose(0, 2, 1)
        if single_output:
            ref = ds.ref_wd.values[:-cutoff][::sample_size]
        else:
            ref = ds.ref_wd.values[:-cutoff].reshape((n_samples, sample_size))

    return trsl, ref


def cnn_classify(
    ds: pd.DataFrame,
    param_dir=str,
    num_channels=2,
    sample_size=60,
    batchsize=256,
    kernel_size=3,
    n_conv_filters=[16, 32, 64, 128],
    n_fc_neurons=64,
    single_output=True,
):
    """
    Classify rainy periods from CML data, using trained cnn modul

    Parameters
    ds : pandas.DataFrame containing CML data and reference rain data for training and testing
    param_dir: str, default = 'default',
    num_channels : int, default = 2, number of variables for cnn to classify from (2*trsl, 2*temperature)
    samplesize : int, default = 100, number of values to be grouped in a sample
    batchsize : int, default = 20, number of samples per batch, to be feed into cnn at once
    kernel_size : int, default = 3
    n_conv_filters : int, number of convolutional layer inputs and outputs
    n_fc_neurons : int, number of neurons in FC layer
    single_output : bool, classification output of the CNN is single value if True, otherwise matches sample_size

    Returns
    cnn_out: np.array containing 0-1 float classification probability output of CNN
    total_loss: float, total cnn prediction loss (w/d reference - cnn output)
    """

    # trsl, ref = get_trsl_and_ref(ds, sample_size, num_channels, single_output)

    # Storing as tensors
    # trsl_data = torch.Tensor(trsl)
    # ref_data = torch.Tensor(ref)

    # Turning into TensorDataset
    dataset = torch.utils.data.TensorDataset(trsl_data, ref_data)

    validloader = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, shuffle=False
    )

    # loading the model parameters:
    path = "cnn_custom_model/"

    model = cnn.cnn_class(
        channels=num_channels,
        sample_size=sample_size,
        kernel_size=kernel_size,
        n_fc_neurons=n_fc_neurons,
        n_filters=n_conv_filters,
        single_output=single_output,
    )

    model.load_state_dict(torch.load(path + param_dir))

    cnn_output = []
    valid_losses = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(validloader):
            pred = model(inputs)
            pred = nn.Flatten(0, 1)(pred)
            cnn_output = cnn_output + pred.tolist()
            loss = nn.BCELoss()(pred, targets)
            valid_losses.append(loss.detach().numpy())
        total_loss = np.mean(valid_losses)

    cnn_out = np.array(cnn_output).reshape(-1)

    return cnn_out, np.mean(total_loss)
