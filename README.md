# Text Summarization Using GPT-2

This project implements text summarization using GPT-2 (Generative Pre-trained Transformer 2) model. The implementation focuses on abstractive summarization, where the model generates concise, coherent summaries of conversations while preserving the key information.

## Project Overview

This project is part of the CS-4063 Natural Language Processing course (Fall 2024) and aims to develop a text summarization system using the GPT-2 model. The system is trained on a conversation dataset to perform abstractive text summarization.

## Dataset

The project uses a conversation summarization dataset that includes:
- Dialogues between two or more participants
- Human-written summaries of these conversations
- Data formatted in instruction-following format for fine-tuning

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- TRL (Transformer Reinforcement Learning)
- Accelerate
- Datasets
- Pandas
- NumPy

## Model Architecture

The project utilizes GPT-2 for abstractive summarization with the following key components:

- Pre-trained GPT-2 model as the base architecture
- Fine-tuning using SFTTrainer (Supervised Fine-Tuning)
- Instruction-following format for better control over generation
- Optimized hyperparameters for conversation summarization

## Evaluation Metrics

The model's performance will be evaluated using:
- Training Loss
- ROUGE Scores
- Summary Quality Assessment
- Human Evaluation of Generated Summaries

## Project Structure

```
├── data/                  # Dataset directory
├── models/               # Saved model checkpoints
├── notebooks/           # Jupyter notebooks for training
│   └── Fine-tuning.ipynb # Main training notebook
├── processed_data/      # Processed dataset files
└── README.md           # Project documentation
```

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd Fine-Tuning-GPT2-Model-for-Text-Summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare the dataset:
- Place the conversation dataset in the `data` directory
- Process the data into the instruction format
- Split into train and validation sets

4. Run the training notebook:
- Open `Fine-tuning.ipynb`
- Follow the steps to fine-tune the GPT-2 model
- Monitor training progress and evaluation metrics

## Usage

To generate a summary for a given conversation:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")

# Format the input
conversation = "your conversation text here..."
input_text = f"### Instruction:\nSummarize the following conversation.\n\n### Input:\n{conversation}\n\n### Summary:"

# Generate summary
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

## Expected Outputs

The model will provide:
- Abstractive summaries of input conversations
- Training metrics and loss values
- Evaluation results on test data

## Future Improvements

- Implementation of beam search and sampling strategies
- Model optimization for longer conversations
- Integration of additional dialogue-specific pre-training
- Web interface for easy testing and demonstration

## License

This project is part of an academic assignment for CS-4063 Natural Language Processing course.

## Contact

For any queries regarding this project, please contact the author:
- Name: Waghib Ahmad
- Email: waghibahmad30@gmail.com
- LinkedIn: https://www.linkedin.com/in/waghibahmad
- GitHub: https://github.com/waghib
- Website: https://waghib.github.io/
