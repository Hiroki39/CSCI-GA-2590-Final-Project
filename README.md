# CSCI-GA 2590 Final Project

## Usage

`python main.py --model <model_name> --dataset <dataset_name> --prompt <prompt method name> --shot <# shots> --dev/--no-dev`

- `model`: `gptturbo` (recommended) or `gpt3`
- `dataset`: `gsm8k` or `aqua`
- `prompt`: `pycot` or `sympy`
- `shot`: `1`, `2`, `4` or `8`
- `dev`: use 5-example mini dataset for debugging or not

Before running the code, please make sure you create a `.env` file in the root directory and add the following line:

`OPENAI_API_KEY=<your openai api key>`
