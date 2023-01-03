#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created By  : Ítalo Della Garza Silva
# Created Date: 02/01/2023
#
# test_nenn_amlsim_tune.py: NENN hyperparametrization
#

import os
import sys
import scipy
import torch
import random
import pickle
import numpy as np

from models.model_nenn import Nenn

from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import RunConfig
from sklearn.metrics import precision_score, recall_score, f1_score

search_space = {
    'node_embed_size': tune.sample_from(lambda _: np.random.randint(3, 12)),
    'edge_embed_size': tune.sample_from(lambda _: np.random.randint(4, 16)),
    'weight': tune.sample_from(lambda _: 0.8 + (np.random.rand()/10.0)),
    'learning_rate': tune.loguniform(1e-3, 1e-5)
}

checkpoint_params = {}
checkpoint = ''
n_repeats = None
output = ''
glob_dataset = None
dataset_name = ''


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


def arg_parser():
    """
    Function to load arguments globally
     """
    global n_repeats, output, dataset_name, checkpoint
    dataset_name = sys.argv[1]
    n_repeats = int(sys.argv[2])
    output = sys.argv[3]
    if len(sys.argv) > 4:
        checkpoint = sys.argv[4]


def data_loader():
    global glob_dataset, checkpoint_params
    if glob_dataset is None:
        dataset = []
        if not checkpoint_params['dataset_order']:
            checkpoint_params['dataset_order'] = os.listdir(dataset_name)
            random.shuffle(checkpoint_params['dataset_order'])
        for ptfile in checkpoint_params['dataset_order']:
            dataset.append(torch.load(f'{dataset_name}/{ptfile}'))

        random.shuffle(dataset)
        glob_dataset = dataset
    # Divide data into train, test and validation.
    train_data = glob_dataset[0:int(0.8 * 366)]
    test_data = glob_dataset[int(0.8 * 366):]
    validation_data = train_data[0:len(test_data)]
    train_data = train_data[len(test_data):]
    return train_data, test_data, validation_data


def checkpoint_loader():
    global checkpoint_params, checkpoint, order
    if checkpoint != '':
        checkpoint = 'nenn_tune'
        with open(f'tmp\\{checkpoint}.txt', 'rb') as file:
            checkpoint_params = pickle.load(file)
    else:
        checkpoint_params['dataset_order'] = []
        checkpoint_params['executions_results'] = {
            'precs_macro': [],
            'recs_macro': [],
            'f1s_macro': [],
            'precs_0': [],
            'recs_0': [],
            'f1s_0': []
        }
        checkpoint_params['best_node_emb_size'] = None
        checkpoint_params['best_edge_emb_size'] = None
        checkpoint_params['best_weight'] = None


def checkpoint_saver():
    with open(f'tmp\\{checkpoint}.txt', 'wb') as file:
        pickle.dump(checkpoint_params, file)


def training_function(config):
    """
        Training function to pass into ray tuning
        """

    # Load data
    train_data, _, validation_data = data_loader()

    f1s = []
    losses = []

    for it in range(5):

        # Model definition
        model = Nenn(
            6,
            8,
            config['node_embed_size'],
            config['edge_embed_size'],
            2
        )
        model = model.to('cpu')

        # Loss definition
        loss = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([1.0-config['weight'], config['weight']])
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
                # Reset gradients
                optimizer.zero_grad()
                # Send info to the model
                logits, _ = model(
                    data.x.T.type(torch.FloatTensor),
                    data.edge_attr.T.type(torch.FloatTensor),
                    data.edge_to_edge_adj_matr.T.type(torch.FloatTensor),
                    data.edge_to_node_adj_matr.T.type(torch.FloatTensor),
                    data.node_to_edge_adj_matr.T.type(torch.FloatTensor),
                    data.node_to_node_adj_matr.T.type(torch.FloatTensor)
                )
                # Calculate loss
                l = loss(logits, data.y)
                l.backward()
                # Update gradients
                optimizer.step()
            print('it = ', it+1, 'epoch =', epoch+1, 'loss =', l.item())

        # Model validation
        label_val_list = []
        y_true_list = []
        val_loss = 0.0
        val_steps = 0

        model.eval()
        with torch.no_grad():
            for data in validation_data:
                data.to('cpu')
                logits, _ = model(
                    data.x.T.type(torch.FloatTensor),
                    data.edge_attr.T.type(torch.FloatTensor),
                    data.edge_to_edge_adj_matr.T.type(torch.FloatTensor),
                    data.edge_to_node_adj_matr.T.type(torch.FloatTensor),
                    data.node_to_edge_adj_matr.T.type(torch.FloatTensor),
                    data.node_to_node_adj_matr.T.type(torch.FloatTensor)
                )
                # Calculate loss
                l_val = loss(logits, data.y)
                val_loss += l_val.numpy()
                val_steps += 1

                # Update true and predicted values.
                label_pred = logits.max(1)[1].tolist()
                label_val_list += label_pred
                y_true_list += data.y.tolist()
            f1_0 = f1_score(
                y_true_list, label_val_list, average='binary', labels=[0]
            )

        f1s.append(f1_0)
        losses.append(val_loss/val_steps)

        # Checkpoint
        os.makedirs("tmp/model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()),
            "tmp/model/checkpoint.pt"
        )
    checkpoint = Checkpoint.from_directory("tmp/model")

    session.report(
        {"loss": np.mean(losses), "f10": np.mean(f1s)},
        checkpoint=checkpoint
    )
    print("Finished Training")


def train_test_best(
    best_node_embed_size,
    best_edge_embed_size,
    best_weight,
    best_lr
):
    """
       Evaluation function to execute with the best results
       """

    # Create the model with the best configuration
    model = Nenn(6, 8, best_node_embed_size, best_edge_embed_size, 2)
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    # Create loss
    loss = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor([1.0 - best_weight, best_weight])
    )

    # Load train and test data
    train_data, test_data, validation_data = data_loader()
    train_data = train_data + validation_data

    label_pred_list = []
    y_true_list = []

    # Training the model
    # Training the model
    model.train()
    for epoch in range(100):
        for ts, data in enumerate(train_data):
            # Reset gradients
            optimizer.zero_grad()
            # Send info to the model
            logits, _ = model(
                data.x.T.type(torch.FloatTensor),
                data.edge_attr.T.type(torch.FloatTensor),
                data.edge_to_edge_adj_matr.T.type(torch.FloatTensor),
                data.edge_to_node_adj_matr.T.type(torch.FloatTensor),
                data.node_to_edge_adj_matr.T.type(torch.FloatTensor),
                data.node_to_node_adj_matr.T.type(torch.FloatTensor)
            )
            # Calculate loss
            l = loss(logits, data.y)
            l.backward()
            # Update gradients
            optimizer.step()
        print('epoch =', epoch + 1, 'loss =', l.item())

    # Model Evaluation
    model.eval()
    with torch.no_grad():
        for data in test_data:
            data.to('cpu')
            logits, _ = model(
                data.x.T.type(torch.FloatTensor),
                data.edge_attr.T.type(torch.FloatTensor),
                data.edge_to_edge_adj_matr.T.type(torch.FloatTensor),
                data.edge_to_node_adj_matr.T.type(torch.FloatTensor),
                data.node_to_edge_adj_matr.T.type(torch.FloatTensor),
                data.node_to_node_adj_matr.T.type(torch.FloatTensor)
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
    global n_repeats, checkpoint, checkpoint_params

    """Main function"""
    if len(sys.argv) <= 3:
        print('Wrong number of arguments')
        print('You must put the 3 necessary arguments:')
        print()
        print('$ test_nenn_amlsim.py <dataset_path_name> ' +
              '<number_of_repetitions> <output_name_file>')
        print()
        sys.exit()

    # Load arguments
    arg_parser()

    # Load checkpoint
    checkpoint_loader()

    # Load dataset
    data_loader()

    print(checkpoint_params['dataset_order'])

    # Configure tuner.
    trainable_with_resources = tune.with_resources(
        training_function, {"cpu": 2}
    )

    tuner = None

    if checkpoint == '':
        tuner = tune.Tuner(
            trainable=trainable_with_resources,
            param_space=search_space,
            run_config=RunConfig(
                name='nenn_tune'
            ),
            tune_config=tune.TuneConfig(
                metric="f10",
                mode="max",
                num_samples=100
            )
        )
    else:
        tuner = tune.Tuner.restore(
            path=f"/home/italo/ray_results/{checkpoint}"
        )

    checkpoint = 'nenn_tune'
    checkpoint_saver()

    # Fit tuner.
    results = tuner.fit()

    if not checkpoint_params['executions_results']['precs_macro']:
        # Fit tuner.
        #results = tuner.fit()

        # Get the best result
        best_result = results.get_best_result("f10", "max")
        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]))
        print("Best trial final validation f10: {}".format(
            best_result.metrics["f10"]))

        print('Optimization finished. Started training with the best '
              +'parameters.')

        best_node_emb_size = best_result.config['node_embed_size']
        best_edge_emb_size = best_result.config['edge_embed_size']
        best_weight = best_result.config['weight']
        best_lr = best_result.config['learning_rate']
        checkpoint_params['best_node_emb_size'] = best_node_emb_size
        checkpoint_params['best_edge_emb_size'] = best_edge_emb_size
        checkpoint_params['best_weight'] = best_weight
        checkpoint_params['best_lr'] = best_lr
        checkpoint_saver()
    else:
        best_node_emb_size = checkpoint_params['best_node_emb_size']
        best_edge_emb_size = checkpoint_params['best_edge_emb_size']
        best_weight = checkpoint_params['best_weight']
        best_lr = checkpoint_params['best_lr']

    # Save results as a Pandas Dataframe
    results.get_dataframe().to_csv(f'results/{output}.csv')

    precs_macro = []
    recs_macro = []
    f1s_macro = []
    precs_0 = []
    recs_0 = []
    f1s_0 = []

    results_passed = len(
        checkpoint_params['executions_results']['precs_macro']
    )
    n_repeats = n_repeats - results_passed

    if results_passed != 0:
        precs_macro = checkpoint_params['executions_results']['precs_macro']
        recs_macro = checkpoint_params['executions_results']['recs_macro']
        f1s_macro = checkpoint_params['executions_results']['f1s_macro']
        precs_0 = checkpoint_params['executions_results']['precs_0']
        recs_0 = checkpoint_params['executions_results']['recs_0']
        f1s_0 = checkpoint_params['executions_results']['f1s_0']

    # Train and Test with the best parameters
    for i in range(n_repeats):
        print(f1s_0)
        print(f'EXECUTION {i + results_passed}')

        prec_macro, rec_macro, f1_macro, prec_0, rec_0, f1_0 = \
        train_test_best(
            best_node_emb_size,
            best_edge_emb_size,
            best_weight,
            best_lr
        )

        print('f1: ', f1_0)

        precs_macro.append(prec_macro)
        recs_macro.append(rec_macro)
        f1s_macro.append(f1_macro)
        precs_0.append(prec_0)
        recs_0.append(rec_0)
        f1s_0.append(f1_0)
        checkpoint_params['executions_results']['precs_macro'] = precs_macro
        checkpoint_params['executions_results']['recs_macro'] = recs_macro
        checkpoint_params['executions_results']['f1s_macro'] = f1s_macro
        checkpoint_params['executions_results']['precs_0'] = precs_0
        checkpoint_params['executions_results']['recs_0'] = recs_0
        checkpoint_params['executions_results']['f1s_0'] = f1s_0
        checkpoint_saver()

    n_repeats = n_repeats + results_passed
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


if __name__ == "__main__":
    main()
