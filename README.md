# Natural Conversation Benchmarks

## Overview

The Natural Conversation Benchmarks (NC-Bench) aim to answer the question: How well can generative AI converse like humans do? In other words, the benchmarks begin to measure the general conversational competence of large language models (LLMs). They do this by testing models' ability to generate an appropriate type of conversational action, or dialogue act, in response to a particular sequence of actions. The sequences of conversational actions, or patterns, are adapted from conversation science, specifically the model of sequence organization in the field of conversation analysis (Schegloff, 2007) and the pattern library of IBM Natural Conversation Framework ([book](https://dl.acm.org/doi/abs/10.1145/3304087)). Models are tested by generating the next line in a transcript. NC-Bench is a lightweight method that is easily extensible to more conversation patterns.

## Natural Conversation Benchmarks

The initial set of benchmarks (Set 1) test how well models can perform three basic conversational actions: providing answers, modifying answers and transitioning to a new sequence. The answering tests include three different ways of completing an inquiry sequence (i.e., producing a 2nd pair part to a base adjacency pair): simple answer, incremental request and self-correction. In the first, the model should generate a straightforward answer to an inquiry. In the second, the model should give a second answer after the user produces a second inquiry (dependent on the first for its meaning) merely by changing a detail from the initial inquiry. And in the third, it should provide an alternative answer after the user corrects the initial inquiry into an alternative inquiry.
The repairing tests include four different types of conversational repairs: repeating, paraphrasing, defining and giving examples. In the first, the model should repeat the agent's prior response using the same words. In the second, the model should paraphrase the agent's prior turn using different, simpler words and phrases. In the third, it should provide a definition of a problematic keyword in the agent's prior response. And in the fourth, it should give an example of what the agent said in a prior response.
Finally, the closing tests include two ways of ending an inquiry sequence: closing and aborting. In the first, the model should acknowledge the user's move to close the sequence instead of continuing to answer. In the second, the model should acknowledge the user's move to cancel the sequence and move on.

## A Benchmark Framework

Grounding our benchmark approach in the [IBM Natural Conversation Framework](NCF) provides extensibility. Not only does the NCF contain over 120 patterns in its library, but its adaptation of the literature and methods of Conversation Analysis mean that additional patterns can be added systematically. The table below contains our roadmap for extending the benchmarks to new patterns.

| Pattern Sets | Answering | Repairing | Closing | Eliciting | Preliminary | Informing |
|-------------|-----------|-----------|---------|-----------|------------|---------|
| **1**       | Inquiry  | Repeat    | Sequence Closer |   |   |   |
|             | Incremental Request | Paraphrase | Sequence Abort |   |   |   |
|             | Self-Correction | Example |   |   |   |   |
|             |   | Definition |   |   |   |   |
| **2**       | Inquiry RAG | Repeat RAG | Sequence Closer RAG |   |   |   |
|             | Inquiry RAG Ungrounded | Paraphrase RAG | Sequence Abort RAG |   |   |   |
|             | Incremental-Self-Correction RAG | Example RAG |   |   |   |   |
|             |   | Definition RAG |   |   |   |   |
| **3**       | Recommendation | Mixed |   | Elicit All | Detail Giving |   |
|             | Expanded Recommendation |   | Elicit Missing | Screening |   |   |
|             | Incremental Recommendation |   | Choices |   |   |   |


Table: Sets of Natural Conversation Benchmark Patterns

Set 1 – Patterns that capture basic practices of sequence management: answering inquiries, repairing answers, and closing pair sequences. This set uses ordinary conversational use cases and does NOT include passages for retrieval augmented generation (RAG). (See results above).

Set 2 – Sequence management patterns from Set 1 but with the inclusion of a passage of information for RAG. Determining the faithfulness of the models' responses to the passage are not a primary goal. Instead, the goal is to determine if the model can maintain the conversation pattern in the face of a document context, which contains a competing language style and format. The uses cases in this set involve information giving using Wikipedia as a source.

Set 3 – Sequence management patterns involving complex requests. Such requests require the agent to elicit details from the fictional user (for example, slot filling). Other patterns involve preliminaries to the inquiry-answer pair (i.e., pre-expansions). These use cases are business related. 

## How to run the benchmark

### 1. **Clone the Repository**

First, clone the repository to your local machine:

```bash
git clone https://github.ibm.com/conversational-ux/conversation-benchmark.git
cd conversation-benchmark
```

### 2. **Install Dependencies**

Create a new conda or Python virtual environment. Install all dependencies using pip:pip:

```bash
pip install -r requirements.txt
```

### 3. **Dataset**

The dataset required for benchmarking can be found in the `data/` directory. There is nothing to do here, the data file will already exist. This is a JSONL file where each entry is a JSON with fields for the ID, task type, and input text. For example:

```json
{
  "id": 0,
  "task": "definition request",
  "chat_prompt": [
        {"role": "system", "content": "Let’s do some roleplaying today! We’re just gonna have an ordinary conversation. But keep your turn short, about one sentence."},
        {"role": "user", "content": "What's the difference between an index fund and a mutual fund?"},
        {"role": "assistant", "content": "Index funds are passively managed, while most other mutual funds are actively managed."},
        {"role": "user", "content": "What does passively managed mean?"}
  ]
}
```

The `chat_prompt` field is the input text formatted to be used for the HuggingFace chat template using the `tokenizer.apply_chat_template(...)` API. The `freeform_prompt` is the raw text without any chat templates that can be used by models that lack a chat template.

### 4. **Run Model Predictions**

To run the model predictions using a local HuggingFace model, use the provided script:

```bash
# Generates outputs using a Hugging Face model for benchmark tasks.
python pred_hf.py --model_path /path/to/model \
        --prompts_file ./data/converation_competence.jsonl \
        --generations_file /path/to/model/generations.jsonl
```

- ```--model_path:``` Path to your model.
- ```--prompts_file:``` Path to the input prompt data (in JSONL format).
- ```--generations_file:``` Path to where the generated outputs will be saved (in JSONL format).
- ```--batch_size:``` Number of samples to process per batch (default is 8).
- ```--max_new_tokens:``` Maximum number of new tokens to generate (default is 128).
- ```--no_chat_template:``` Flag to disable the chat template (off by default).

To run the model predictions using an OpenAI model, use the provided script:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="..."

# Generates outputs using GPT-4o for benchmark tasks.
python pred_gpt4o.py --model gpt-4o-2024-08-06 \
        --prompts_file ./data/converation_competence.jsonl \
        --generations_file /path/to/model/generations.jsonl
```

- ```--model:``` Specify the GPT-4o model version to use (default is gpt-4o-2024-08-06).
- ```--prompts_file:``` Path to the input prompt data (in JSONL format).
- ```--generations_file:``` Path to where the generated outputs will be saved (in JSONL format).
- ```--max_new_tokens:``` Maximum number of new tokens to generate (default is 128).

To run the model predictions [Research Inference and Training Services (RITS)](https://rits.fmaas.res.ibm.com/), use the provided script:

```bash
# Set RITS API key
export RITS_API_KEY="..."

# Generates outputs using RITS for benchmark tasks.
python pred_rits.py --model ibm-granite/granite-13b-chat-v2 \
        --endpoint https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-13b-chat-v2 \
        --prompts_file ./data/converation_competence.jsonl \
        --generations_file /path/to/model/generations.jsonl \
```

- ```--model:``` Specify the model version to use.
- ```--endpoint:``` Model inference endpoint.
- ```--prompts_file:``` Path to where the input data is located (in JSONL format).
- ```--generations_file:``` Path to where the generated outputs will be saved (in JSONL format).
- ```--max_new_tokens:``` Maximum number of new tokens to generate (default is 128).

### 5. **Run Judgments using an LLM as a Judge**

To run the judgments on the model generations with a local HuggingFace model, use the provided script:

```bash
# Evaluates the generated outputs using a Hugging Face model as the judge.
python judge_hf.py --judge_path /path/to/judge \
        --prompts_file ./data/converation_competence.jsonl \
        --generations_file /path/to/model/generations.jsonl \
        --judgments_file /path/to/judgments.jsonl
```

- ```--judge_path:``` Path to the model (e.g. Llama-3.3-70b-Instruct).
- ```--prompts_file:``` Path to the input prompt data (in JSONL format).
- ```--generations_file:``` Path to the generated outputs (in JSONL format).
- ```--judgments_file:``` Path where the judgment outputs will be saved (in JSONL format).
- ```--max_new_tokens:``` Maximum number of new tokens to evaluate (default is 128).
- ```--batch_size:``` Number of samples to process per batch (default is 4).

To run the judgments on the model generations with an OpenAI model, use the provided script:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="..."

# Evaluates the generated outputs using GPT-4o as the judge.
python judge_gpt4o.py --judge gpt-4o-2024-08-06 \
        --prompts_file ./data/converation_competence.jsonl \
        --generations_file /path/to/model/generations.jsonl \
        --judgments_file /path/to/judgments.jsonl
```

- ```--judge:``` Specify the GPT-4o model version to use (default is gpt-4o-2024-08-06).
- ```--prompts_file:``` Path to the input prompt data (in JSONL format).
- ```--generations_file:``` Path to the generated outputs (in JSONL format).
- ```--judgments_file:``` Path where the judgment outputs will be saved (in JSONL format).
- ```--max_new_tokens:``` Maximum number of new tokens to evaluate (default is 128).

### 6. **Aggregate the Scores**

After the evaluation process is complete, aggregate the results by running the provided script. This will produce an output file containing the chosen metric(s).

- **Input:** The judgments file in JSONL format, such as `judgments/gpt-4o/llama-3.2-1b-instruct.jsonl`.
- **Output:** The aggregated results saved in a JSON file, such as `evaluations/gpt-4o/llama-3.2-1b-instruct.json`.

Note: The aggregated output should be saved in **JSON format** rather than JSONL since it contains the final aggregated metric(s).

To run the aggregation process, use the following command:

```bash
python aggregator.py --judgments_dir ./results/judgments/gpt-4o --evaluations_dir ./results/evaluations/gpt-4o
```

- ```--judgments_dir:``` The path to the directory containing the judgments.
- ```--evaluations_dir:``` The path to the directory where the aggregated results (in JSON format) will be saved.

## Citation

[Emanuel A. Schegloff, *Sequence Organization in Interaction: A Primer in Conversation Analysis* (Cambridge University Press, 2007)](https://www.cambridge.org/core/books/sequence-organization-in-interaction/276CD30E23D3444114A90E5E2B24D55F)
