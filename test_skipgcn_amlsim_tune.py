#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created By  : Ítalo Della Garza Silva
# Created Date: 02/01/2023
#
# test_skipgcn_amlsim_tune.py: Skip-GCN hyperparametrization
#

import os
import sys
import torch
import random
import numpy as np
import scipy
from models.model_skipgcn import SkipGCN
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint


search_space = {
    'neurons_l1': tune.sample_from(lambda _: np.random.randint(8, 28)),
    'weight_1': tune.sample_from(lambda _: 0.6 + (np.random.rand()/2.5)),
    'learning_rate': tune.loguniform(1e-2, 1e-4)
}

# TODO ###################### Recuperar O treinamento 
n_repeats = None
output = ''
dataset_name = ''
glob_dataset = None


def get_confidence_intervals(metric_list, n_repeats):
    """Function to calculate the confidence intervals with a 95%
    confidence value.

    :param metric_list: list containing the metrics obtained.
    :type metric_list: list
    :param n_repeats: number of experiment repetitions.
    :type n_repeats: int

    :return: (metric average, confidence interval length)
    :rtype: (float, float)
    """
    confidence = 0.95
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_repeats - 1)
    metric_avg = np.mean(metric_list)

    se = 0.0
    for m in metric_list:
        se += (m - metric_avg) ** 2
    se = np.sqrt((1.0 / (n_repeats - 1)) * se)
    ci_length = t_value * se

    return metric_avg, ci_length




def arg_loader():
    """
    Function to load arguments globally
    """
    global n_repeats, output, dataset_name, checkpoint
    dataset_name = sys.argv[1]
    n_repeats = int(sys.argv[2])
    output = sys.argv[3]



def data_loader():
    """
    Function to load global data, if not loaded
    """
    global glob_dataset, n_repeats, output, dataset_name
    if glob_dataset is None:
        dataset_name = sys.argv[1]
        glob_dataset = []
        for ptfile in os.listdir(dataset_name):
            glob_dataset.append(torch.load(f'{dataset_name}/{ptfile}'))
        new_dataset = []

        # Dataset conversion to let the transactions be the graph nodes
        for data in glob_dataset:
            edge_index = data.edge_to_edge_adj_matr - \
                torch.eye(data.edge_to_edge_adj_matr.shape[0])
            edge_index = edge_index + edge_index.T
            edge_index = edge_index.nonzero()
            x = data.edge_attr
            y = data.y
            new_data = Data(
                x=x,
                y=y,
                edge_index=edge_index
            )
            new_dataset.append(new_data)
        random.shuffle(new_dataset)
        glob_dataset = new_dataset

    # Divide data into train, test and validation.
    train_data = glob_dataset[0:int(0.8 * 366)]
    test_data = glob_dataset[int(0.8 * 366):]
    validation_data = train_data[0:len(test_data)]
    train_data = train_data[len(test_data):]
    return train_data, test_data, validation_data



def training_function(config):
    """
    Training function to pass into ray tuning
    """

    # Load data
    train_data, _, validation_data = data_loader()

    f1s = []
    losses = []

    for _ in range(5):
        # Model definition
        model = SkipGCN(8, config['neurons_l1'], 2)
        model = model.to('cpu')

        # Loss definition
        loss = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([1.0-config['weight_1'], config['weight_1']])
        )

        # Optimizer definition
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )

        # Restore checkpoint
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(loaded_checkpoint_dir, 'checkpoint.pt')
                )
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

        # Training the model
        model.train()
        for epoch in range(100):
            for ts, data in enumerate(train_data):
                data.to('cpu')
                optimizer.zero_grad()
                hidden, logits = model(
                    data.x.T.float(), data.edge_index.T, None
                )
                l = loss(logits, data.y.T)
                l.backward()
                optimizer.step()
            print('epoch =', epoch + 1, 'loss =', l.item())

        # Model validation
        label_pred_list_val = []
        y_true_list_val = []
        val_loss = 0.0
        val_steps = 0

        model.eval()
        with torch.no_grad():
            for val_data in validation_data:
                val_data.to('cpu')
                _, val_logits = model(
                    val_data.x.T.float(), val_data.edge_index.T, None
                )
                l_val = loss(val_logits, val_data.y.T)

                val_loss += l_val.numpy()
                val_steps += 1

                label_pred_val = val_logits.max(1)[1].tolist()
                label_pred_list_val += label_pred_val
                y_true_list_val += val_data.y.tolist()
            model.train()
            label_pred_list_val = np.array(label_pred_list_val)
            y_true_list_val = np.array(y_true_list_val)
            f1_0 = f1_score(
                y_true_list_val,
                label_pred_list_val,
                average='binary',
                labels=[0]
            )
        f1s.append(f1_0)
        losses.append(val_loss/val_steps)
        # Checkpoint
        os.makedirs("tmp/model", exist_ok=True)
        torch.save(
            (
                model.state_dict(),
                optimizer.state_dict()
            ),
            "tmp/model/checkpoint.pt"
        )
    checkpoint = Checkpoint.from_directory("tmp/model")
    session.report(
        {"loss": np.mean(losses), "f10": np.mean(f1s)},
        checkpoint=checkpoint
    )
    print("Finished Training")


def train_test_best(best_neurons_l1, best_weight_1, best_lr):
    """
    Evaluation function to execute with the best results
    """

    # Create the model with the best configuration
    model = SkipGCN(8, best_neurons_l1, 2)
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    # Create loss
    loss = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor([1.0 - best_weight_1, best_weight_1])
    )

    # Load checkpoint state to the best model
    #checkpoint_path = os.path.join(
    #    best_results.checkpoint.to_directory(), 'checkpoint.pt'
    #)
    #_, optimizer_state = torch.load(checkpoint_path)
    #model.load_state_dict(model_state)
    #optimizer.load_state_dict(optimizer_state)

    # Load train and test data
    train_data, test_data, val_data = data_loader()
    train_data = train_data + val_data


    label_pred_list = []
    y_true_list = []

    # Training the model
    model.train()
    for epoch in range(100):
        for ts, data in enumerate(train_data):
            data.to('cpu')
            optimizer.zero_grad()
            hidden, logits = model(
                data.x.T.float(), data.edge_index.T, None
            )
            l = loss(logits, data.y.T)
            l.backward()
            optimizer.step()
        print('epoch =', epoch + 1, 'loss =', l.item())

    # Model evaluation
    model.eval()
    with torch.no_grad():
        for data in test_data:
            data.to('cpu')
            _, logits = model(
                data.x.T.float(), data.edge_index.T, None
            )
            label_pred = logits.max(1)[1].tolist()
            label_pred_list += label_pred
            y_true_list += data.y.tolist()
    model.train()
    prec_macro = precision_score(
        y_true_list, label_pred_list, average='macro'
    )
    rec_macro = recall_score(
        y_true_list, label_pred_list, average='macro'
    )
    f1_macro = f1_score(y_true_list, label_pred_list, average='macro')

    prec_0 = precision_score(
        y_true_list, label_pred_list, average='binary', labels=[0]
    )
    rec_0 = recall_score(
        y_true_list, label_pred_list, average='binary', labels=[0]
    )
    f1_0 = f1_score(
        y_true_list, label_pred_list, average='binary', labels=[0]
    )

    del model

    return prec_macro, rec_macro, f1_macro, prec_0, rec_0, f1_0



def main():
    """Main function"""
    if len(sys.argv) <= 3:
        print('Wrong number of arguments')
        print('You must put the 3 necessary arguments:')
        print()
        print('$ test_gcn_amlsim.py <dataset_path_name> ' +
              '<number_of_repetitions> <output_name_file>')
        print()
        sys.exit()
    # Load arguments
    arg_loader()

    # Load dataset
    data_loader()
    
    # Create dir to save checkpoints
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')

    # Configure tuner.
    trainable_with_resources = tune.with_resources(
        training_function, {"cpu": 2}
    )
    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="f10",
            mode="max",
            num_samples=100,
        )
    )

    # Fit tuner
    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result("f10", "max")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation f10: {}".format(
        best_result.metrics["f10"]))
    
    
    precs_macro = []
    recs_macro = []
    f1s_macro = []
    precs_0 = []
    recs_0 = []
    f1s_0 = []
    
    print('Optimization finished. Started training with the best parameters.')

    best_neurons_l1 = best_result.config['neurons_l1']
    best_weight_1 = best_result.config['weight_1']
    best_lr = best_result.config['learning_rate']

    # Save results as a Pandas Dataframe
    results.get_dataframe().to_csv(f'results/{output}.csv')

    # Train and Test with the best parameters
    for i in range(n_repeats):
        print(f'EXECUTION {i}')
        prec_macro, rec_macro, f1_macro, prec_0, rec_0, f1_0 \
            = train_test_best(best_neurons_l1, best_weight_1, best_lr)
        precs_macro.append(prec_macro)
        recs_macro.append(rec_macro)
        f1s_macro.append(f1_macro)
        precs_0.append(prec_0)
        recs_0.append(rec_0)
        f1s_0.append(f1_0)

    # Calculate confidence intervals
    result = ""
    metric, ci = get_confidence_intervals(precs_macro, n_repeats)
    result += f"Macro Precision: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(recs_macro, n_repeats)
    result += f"Macro Recall: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(f1s_macro, n_repeats)
    result += f"Macro F1: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(precs_0, n_repeats)
    result += f"Ilicit Precision: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(recs_0, n_repeats)
    result += f"Ilicit Recall: {metric} +- {ci}\n"
    metric, ci = get_confidence_intervals(f1s_0, n_repeats)
    result += f"Ilicit F1: {metric} +- {ci}\n"
    print(result)
    
    if not os.path.exists('results'):
        os.mkdir('results')

    with open(f'results/{output}.txt', 'w') as file:
        file.write(result)
        file.close()





if __name__ == '__main__':
    main()

