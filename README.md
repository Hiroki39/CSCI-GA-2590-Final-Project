# CSCI-GA 2590 Final Project

## Usage

`python main.py --model <model_name> --dataset <dataset_name> --prompt <prompt method name> --shot <# shots> --dev/--no-dev`

- `model`: `gptturbo` (recommended), `gpt3`, or `text-davinci-002`
- `dataset`: `gsm8k`, `aqua`, or `multiarith`
- `prompt`: `pycot`, `sympy` or `arithcot`
- `shot`: `1`, `2`, `4` or `8`
- `dev`: use 5-example mini dataset for debugging or not
- `name`: name of the output file (default to log_files.csv)
- `promptset`: set of prompt used (default to the name of dataset)

Before running the code, please make sure you create a `.env` file in the root directory and add the following line:

`OPENAI_API_KEY=<your openai api key>`
