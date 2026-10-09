
# Source Code Credits and Acknowledgments

## Project
University of the Cumberlands Student Support Chatbot

University of the Cumberlands
MSAI-631: Artificial Intelligence for Human-Computer Interaction

## Reused Tutorial Code

### Build AI Chatbot in 5 Minutes with Hugging Face and Gradio
Author: Abid Ali Awan, KDnuggets (June 30, 2023)
Source: https://www.kdnuggets.com/2023/06/build-ai-chatbot-5-minutes-hugging-face-gradio.html

The tutorial's approach was used as the starting point: loading a
Hugging Face model and tokenizer with Transformers, wrapping a
response function in a Gradio interface, and hosting the app on
Hugging Face Spaces.

The group extended the tutorial as follows:

- Replaced the DialoGPT conversation model with the
  Qwen2.5-0.5B-Instruct instruction-following model
- Replaced the older gr.Interface and state approach with
  gr.ChatInterface and its built-in chat history
- Added retrieval-augmented generation (RAG): TF-IDF and cosine
  similarity select relevant University of the Cumberlands
  information for each question
- Added a University of the Cumberlands knowledge file with
  official offices and links
- Added course schedule files so deadlines are only given when
  they are known
- Added a system prompt that limits answers to the provided
  information, protects personal information, and declines
  unrelated questions
- Added a code-level fact-check that replaces answers containing
  links, contacts, or dates not found in the knowledge files

## Earlier Project Version

Version 1 of the project (Flask, HTML, CSS, JavaScript, and TF-IDF
question matching) was written by the group. Its TF-IDF and cosine
similarity approach was reused for the retrieval step.

## AI-Assisted Development

OpenAI ChatGPT was used to assist with:

- Generating initial Python, HTML, CSS, and JavaScript code for
  version 1
- Troubleshooting file placement and import errors
- Explaining Flask and Python environment configuration
- Preparing setup instructions and documentation

Anthropic Claude (Claude Code) was used to assist with:

- Converting the project to Gradio and a Hugging Face model
- Writing the retrieval step and system prompt
- Collecting official University of the Cumberlands links
- Updating setup instructions and documentation

Generated code was reviewed, adapted, and tested locally.

## Language Model

### Qwen2.5-0.5B-Instruct (default) and Qwen2.5-1.5B-Instruct (optional)
Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
License: Apache 2.0

Open-source instruction model by the Qwen team (Alibaba Cloud),
used to generate the chatbot's answers.

## Third-Party Libraries

### Gradio
Website: https://www.gradio.app/

Provides the chat interface and web server.

### Hugging Face Transformers
Website: https://huggingface.co/docs/transformers

Downloads and runs the language model.

### PyTorch
Website: https://pytorch.org/

Runs the model computations used by Transformers.

### Scikit-learn
Website: https://scikit-learn.org/

Provides the TF-IDF vectorizer and cosine similarity used to find
relevant information for each question.

## University Information

The information in cumberlands_info.md was collected from official
University of the Cumberlands web pages (https://www.ucumberlands.edu).
Each source link is listed in the file.

## Additional Technologies

Python: https://www.python.org/

Hugging Face Spaces: https://huggingface.co/spaces

Visual Studio Code: https://code.visualstudio.com/

GitHub: https://github.com/

## External Code Reuse

Apart from the tutorial above, software libraries, and AI-assisted
code generation, the project does not identify any directly
copied third-party code.

If additional external code is incorporated, its author, source
URL, license, and any modifications must be documented here.

## Academic Integrity

The group is responsible for reviewing, testing, and explaining
the submitted code.

The chatbot is an academic prototype and is not an official
University of the Cumberlands support service.
