# DiffusionDialoguePlanning
Implementation of the paper "Planning with Diffusion Models for Target-Oriented Dialogue Systems"

# Environments
The environment requirements are in the file `environment.yml`.

# Download SEDD-medium checkpoint
`python load_model.py`

# Train the Diffusion Language Model
`python train.py`

To customize training configurations or datasets, edit the settings in `configs/config.yaml`.

If you encounter a GPU out-of-memory issue during training, try reducing the batch size and increasing the number of gradient accumulation steps in the training configurations.

After training is finished, the model checkpoint should be located in `./exp_local/<dataset_name>/<experiment_date>/<experiment_number>`.

# Sample Dialogue Plans

## PersonaChat
`mkdir PersonaChat_test` to create a directory for the dialogue plans.

`python run_sample_personachat.py --model_path <model_path> --file_to_save ./PersonaChat_test/PersonaChat_test_<1/2/3/ ...>.json`

The `--model_path` should be located in `./exp_local/PersonaChat/<experiment_date>/<experiment_number>/`.

To sample multiple dialogue plans for MBR decoding, use different file names for `--file_to_save`, such as `PersonaChat_test_1.json`, `PersonaChat_test_2.json` and `PersonaChat_test_3.json`.

## TopDial
`python run_sample_topdial.py --model_path <model_path>` 

The `--model_path` should be located in `./exp_local/TopDial/<experiment_date>/<experiment_number>/`.

## CraigslistBargain
For the CraigslistBargain dataset, the dialogue plan (action strategies) can be generated dynamically during the converation using search-based guidance (see below).

# Dynamic Dialogue Evaluation

To run dynamic dialogue evaluation, you may need to set your OpenAI Key and Anthropic API Key in these files.

## PersonaChat
`python PersonaChat_conversation.py`

## TopDial
`python topdial_conversation.py`

## CraigslistBargain
`python cb_buyer_conversation.py --model_path <model_path>`

`python cb_seller_conversation.py --model_path <model_path>`

The `--model_path` should be located in `./exp_local/cb/<experiment_date>/<experiment_number>/`.

# Calculate reference-based metrics for the PersonaChat and TopDial dataset

## PersonaChat
`python evaluate_PersonaChat.py`

For the PersonaChat dataset, you may sample multiple dialogue plans for MBR decoding. To achieve this, make sure that your dialogue plan file is saved as PersonaChat_test_<1/2/3/ ...>.json, and use --num to set the number of guidances.

## TopDial
`python evaluate_topdial.py`

For the TopDial dataset, you may use --semantic_guidance_number to set the number of guidances for MBR decoding.
