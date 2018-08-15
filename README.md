# Adversarial Relation Classification

Creating large datasets for biomedical relation classification can be prohibitively expensive. While some datasets have been curated to extract protein-protein and drug-drug interactions from text, we are also interested in other interactions including gene-disease and chemical-protein connections. Also, many biomedical researchers have begun to explore ternary relationships. Even when annotated data is available, many datasets used for relation classification are inherently biased. For example, issues such as sample selection bias typically prevent models from generalizing in the wild. To address the problem of cross-corpora generalization, in this repository, we host code for a novel adversarial learning algorithm for unsupervised domain adaptation tasks where no labeled data is available for the target domain. Instead, our method takes advantage of unlabeled data to improve biased classifiers by learning domain-invariant features via a 2-stage adversarial training process using neural networks.

**Note:** Examples of the data format can be found in the data/ directory and the source code for the models are available in the models/ directory.

## Usage
### Training

Below we display an example command-line configuration to train a new model from scratch:

```
python train_final_cnn.py --num_epochs 50 --checkpoint_dir /checkpoint/dir/experiments/checkpoints/ --checkpoint_name my_checkpoint --min_df 5 --lr 0.001 --penalty 0. --adv_train_data_X  /my/data/data1/all_train.txt --adv_test_data_X  /my/data/biogrid_train_test/all_test.txt --test_data /my/data/test_data.txt --train_data /my/data/train_data.txt --train_data_X /my/data/data2/train.txt --val_data_X /my/data/data2/test.txt --num_iters 10000 --num_disc_updates 1 --emb_reg --adv --pos_reg --hidden_state 128 --adv --seed 42
```

The available command-line arguments are listed below:

```
usage: train_final_cnn.py [-h] [--num_epochs NUM_EPOCHS]
                          [--hidden_state HIDDEN_STATE]
                          [--checkpoint_dir CHECKPOINT_DIR]
                          [--checkpoint_name CHECKPOINT_NAME]
                          [--min_df MIN_DF] [--lr LR] [--penalty PENALTY]
                          [--train_data_X TRAIN_DATA_X]
                          [--train_data TRAIN_DATA] [--test_data TEST_DATA]
                          [--val_data_X VAL_DATA_X]
                          [--adv_train_data_X ADV_TRAIN_DATA_X]
                          [--adv_test_data_X ADV_TEST_DATA_X]
                          [--num_iters NUM_ITERS] [--grad_clip GRAD_CLIP]
                          [--num_disc_updates NUM_DISC_UPDATES] [--seed SEED]
                          [--adv] [--emb_reg] [--pos_reg]

Train Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of updates to make.
  --hidden_state HIDDEN_STATE
                        LSTM hidden state size.
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoint directory.
  --checkpoint_name CHECKPOINT_NAME
                        Checkpoint File Name.
  --min_df MIN_DF       Min word count.
  --lr LR               Learning Rate.
  --penalty PENALTY     Regularization Parameter.
  --train_data_X TRAIN_DATA_X
                        Training Data.
  --train_data TRAIN_DATA
                        Training Data.
  --test_data TEST_DATA
                        Training Data.
  --val_data_X VAL_DATA_X
                        Validation Data.
  --adv_train_data_X ADV_TRAIN_DATA_X
                        Validation Data.
  --adv_test_data_X ADV_TEST_DATA_X
                        Validation Data.
  --num_iters NUM_ITERS
                        Validation Data.
  --grad_clip GRAD_CLIP
                        Gradient Clip Value.
  --num_disc_updates NUM_DISC_UPDATES
                        Number of time to update discriminator.
  --seed SEED           Random seed.
  --adv                 Adversarial training?
  --emb_reg             Regularize word embeddings?
  --pos_reg             Regularize pos embeddings?
```

``val_data`` should contain a list of IDs in the source dataset to use to avoid over-fitting during stage 1 of the training process. We provide examples of the format and structure of the train, dev, and test datasets in the data/ directory. Also, note that while we checkpoint the model using the ``checkpoint_dir`` and ``checkpoint_name`` parameters, after stage 2 of the training process the provided source code will also make predictions on the provided test data if the ``test_data_X`` (the target test set) is used.

## Acknowledgements

If you find this useful or relevant in your research, please cite the following paper:

> Anthony Rios, Ramakanth Kavuluru, and Zhiyong Lu. "Generalizing Biomedical Relation Classification with Neural Adversarial Domain Adaptation". Bioinformatics 2018

```
@article{rios2018advrel,
  title={Generalizing Biomedical Relation Classification with Neural Adversarial Domain Adaptation},
  author={Rios, Anthony and Kavuluru, Ramakanth and Lu, Zhiyong},
  journal={Bioinformatics},
  year={2018}
}
```

Written by Anthony Rios (anthonymrios at gmail dot com)
