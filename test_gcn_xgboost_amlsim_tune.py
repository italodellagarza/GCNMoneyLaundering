#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created By  : Ítalo Della Garza Silva
# Created Date: 02/01/2023
#
# test_gcn_xgboost_amlsim_tune.py: GCN + XGBoost hyperparametrization
#

import os
import sys
import torch
import random
import numpy as np
import scipy
import xgboost as xgb
from models.model_gcn import GCN
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint

search_space = {
    'neurons_l1': tune.sample_from(lambda _: np.random.randint(8, 28)),
    'weight_1': tune.sample_from(lambda _: 0.6 + (np.random.rand()/2.5)),
    'learning_rate': tune.loguniform(1e-2, 1e-4),
    'eta': tune.loguniform(0.01, 0.2),
    "max_depth": tune.sample_from(lambda _: np.random.randint(3, 10))
}


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
        model = GCN(8, config['neurons_l1'], 2)
        model = model.to('cpu')

        # Loss definition
        loss = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([1.0 - config['weight_1'], config['weight_1']])
        )

        # Optimizer definition
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Restore checkpoint
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, 'checkpoint.pt'))
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

        # Obtain embeddings to train xgboost
        embeddings_train = []
        y_train = []

        model.eval()
        with torch.no_grad():

            for data in train_data:
                data.to('cpu')
                embedding_train, _ = model(
                    data.x.T.float(), data.edge_index.T, None
                )
                embeddings_train += embedding_train.numpy().tolist()
                y_train += data.y.numpy().tolist()
        model.train()

        embeddings_train = np.array(embeddings_train)
        y_train = np.array(y_train)

        # Creating and training the model
        print('Training XGBoost...')
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=42,
            eta=config['eta'],
            max_depth=config['max_depth']
        )
        xgb_model.fit(embeddings_train, y_train)

        # Obtain embeddings to validade xgboost
        embeddings_val = []
        y_true = []
        val_loss = 0.0
        val_steps = 0

        model.eval()
        with torch.no_grad():

            for data in validation_data:
                data.to('cpu')
                embedding_val, val_logits = model(
                    data.x.T.float(), data.edge_index.T, None
                )
                l_val = loss(val_logits, data.y.T)
                val_loss += l_val.numpy()
                val_steps += 1
                embeddings_val += embedding_val.numpy().tolist()
                y_true += data.y.numpy().tolist()
        model.train()

        embeddings_val = np.array(embeddings_val)
        y_true = np.array(y_true)

        # Validation
        y_val = xgb_model.predict(embeddings_val)

        f1_0 = f1_score(
            y_true, y_val, average='binary', labels=[0]
        )
        f1s.append(f1_0)
        losses.append(val_loss/val_steps)

        # Checkpoint
        os.makedirs("tmp/model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "tmp/model/checkpoint.pt")
    checkpoint = Checkpoint.from_directory("tmp/model")
    session.report({"loss": (val_loss / val_steps), "f10": f1_0}, checkpoint=checkpoint)
    print("Finished Training")


def train_test_best(best_results):
    """
    Evaluation function to execute with the best results
    """

    # Create the model with the best configuration
    model = GCN(8, best_results.config['neurons_l1'], 2)
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=best_results.config['learning_rate'])
    # Create loss
    loss = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor([1.0 - best_results.config['weight_1'], best_results.config['weight_1']])
    )

    # Load checkpoint state to the best model
    checkpoint_path = os.path.join(best_results.checkpoint.to_directory(), 'checkpoint.pt')
    _, optimizer_state = torch.load(checkpoint_path)
    # model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    # Load train and test data
    train_data, test_data, validation_data = data_loader()
    train_data = train_data + validation_data

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

    # Obtain embeddings to train xgboost
    embeddings_train = []
    y_train = []

    model.eval()
    with torch.no_grad():

        for data in train_data:
            data.to('cpu')
            embedding_train, _ = model(
                data.x.T.float(), data.edge_index.T, None
            )
            embeddings_train += embedding_train.numpy().tolist()
            y_train += data.y.numpy().tolist()
    model.train()

    embeddings_train = np.array(embeddings_train)
    y_train = np.array(y_train)

    # Creating and training the model
    print('Training XGBoost...')
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eta=best_results.config['eta'],
        max_depth=best_results.config['max_depth']
    )
    xgb_model.fit(embeddings_train, y_train)

    # Obtain embeddings to evaluate xgboost
    embeddings_test = []
    y_true_list = []

    model.eval()
    with torch.no_grad():

        for data in test_data:
            data.to('cpu')
            embedding_test, _ = model(
                data.x.T.float(), data.edge_index.T, None
            )
            embeddings_test += embedding_test.numpy().tolist()
            y_true_list += data.y.numpy().tolist()
    model.train()

    embeddings_test = np.array(embeddings_test)
    y_true_list = np.array(y_true_list)

    # Test
    label_pred_list = xgb_model.predict(embeddings_test)

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
    trainable_with_resources = tune.with_resources(training_function, {"cpu": 2})
    
    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="f10",
            mode="max",
            num_samples=200
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

    print('Optimization finished. Started training with the best parameters.')

    precs_macro = []
    recs_macro = []
    f1s_macro = []
    precs_0 = []
    recs_0 = []
    f1s_0 = []
    

    # Save results as a Pandas Dataframe
    results.get_dataframe().to_csv(f'results/{output}.csv')
    
    # Train and Test with the best parameters
    for i in range(n_repeats):
        print(f'EXECUTION {i}')
        prec_macro, rec_macro, f1_macro, prec_0, rec_0, f1_0 \
            = train_test_best(best_result)
        precs_macro.append(prec_macro)
        recs_macro.append(rec_macro)
        f1s_macro.append(f1_macro)
        precs_0.append(prec_0)
        recs_0.append(rec_0)
        f1s_0.append(f1_0)
        print(f'F1 il: {f1_0}')

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

