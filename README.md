---
title: UC Student Support Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# University of the Cumberlands Student Support Chatbot

- Live demo (Hugging Face Space): https://huggingface.co/spaces/MRR24/uc-student-support-chatbot
- Source code (GitHub): https://github.com/rohith4444/University-Student-Support-Chatbot

## Project Description

The University of the Cumberlands Student Support Chatbot is a
Gradio web application that helps University of the Cumberlands
students find routine academic and administrative information.

It answers questions about assignment deadlines, academic
expectations, academic advising, course registration, financial
aid, library services, IT support, university tools, and student
support services, and points students to official university
resources.

The chatbot runs a free, open-source language model
(Qwen2.5-0.5B-Instruct) from Hugging Face. No paid API or API key
is needed. It runs on a typical laptop and can be hosted for free
on Hugging Face Spaces.

This is a student project for MSAI 631. It is not an official
University of the Cumberlands service.

## How the Chatbot Works

The chatbot uses retrieval-augmented generation (RAG):

1. The student asks a question in the Gradio chat window.
2. TF-IDF and cosine similarity (scikit-learn) find the three most
   relevant sections in the knowledge files:
   - cumberlands_info.md: official University of the Cumberlands
     offices, services, and links
   - course_schedules/: assignment due dates for added courses
3. Those sections, a system prompt, and the recent chat history are
   sent to the Qwen language model, which runs locally.
4. The model writes a short answer based only on that information.
5. The answer is fact-checked in code before it is shown.

The system prompt tells the model to:

- Share only links and contacts that appear in the knowledge files
- Give an assignment due date only if it is in a course schedule;
  otherwise, direct the student to the course page and syllabus
- Never ask for passwords, student IDs, or personal information
- Politely decline questions unrelated to student support

Because small models do not always follow instructions, the code
also enforces these rules:

- Fact-check: every link, email, phone number, and dollar amount in
  the answer must appear in the knowledge files, and every date must
  belong to the assignment and course the student asked about.
  Otherwise, the answer is replaced with a message pointing to the
  most relevant official office.
- If no related university information is found, the chatbot
  replies that it can only help with student questions.
- If a message seems to contain a password or student ID, the
  chatbot warns the student not to share it.
- Answers that hit the length limit are trimmed to the last full
  sentence.

## Technologies Used

- Python
- Gradio (chat interface)
- Hugging Face Transformers and PyTorch (runs the language model)
- Qwen2.5-0.5B-Instruct (open-source language model, Apache 2.0)
- Scikit-learn (TF-IDF and cosine similarity for retrieval)
- Hugging Face Spaces with ZeroGPU (free hosting) and the spaces package

## System Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer (tested with Python 3.11)
- About 8 GB of RAM
- About 5 GB of free disk space (packages and model)
- Internet connection for the first run (to download packages and
  the model); after that, it runs offline
- A modern web browser

## Installation Instructions

1. Download or clone this project.
2. Open a terminal in the main project folder.

3. Create a Python virtual environment:

   python -m venv .venv

4. Activate the environment on Windows CMD:

   .venv\Scripts\activate.bat

   On macOS or Linux:

   source .venv/bin/activate

5. Install the dependencies:

   python -m pip install -r requirements.txt

6. Start the application:

   python app.py

   The first start downloads the model (about 1 GB) and can take
   a few minutes.

7. Open the browser and visit:

   http://127.0.0.1:7860

Press Ctrl+C in the terminal to stop the application.

On a fast laptop, you can open chatbot.py and change MODEL_NAME to
"Qwen/Qwen2.5-1.5B-Instruct" for better-worded but slower answers
(about 3 GB download; roughly 25-45 seconds per answer in testing,
compared with roughly 6-20 seconds for the default model).

## Running on Hugging Face Spaces

1. Create a free account at https://huggingface.co.
2. Select New Space, choose Gradio as the SDK and ZeroGPU
   hardware (free accounts can host up to 2 ZeroGPU Spaces).
3. Upload app.py, chatbot.py, requirements.txt, README.md,
   CREDITS.md, cumberlands_info.md, and the course_schedules folder.
4. The Space builds automatically and shows the chatbot when ready.

ZeroGPU requires a supported PyTorch version (2.13.0 is used here)
and the spaces package. The @spaces.GPU decorator in chatbot.py
requests a GPU only while an answer is generated, and has no effect
when the app runs on a laptop.

## How to Use the Chatbot

1. Open the chatbot page.
2. Type a question, or click one of the example questions.
3. Press Enter and read the answer.
4. Ask follow-up questions as needed.

## Example Questions

- How do I contact financial aid?
- When is the MSAI 631 results report due?
- How do I reset my password?
- How do I access library databases?
- Who can help me plan my courses?

## Updating the Knowledge

- To update university information, edit cumberlands_info.md.
  Each office is a section starting with "## ".
- To add a course schedule, add a file to course_schedules/
  with one "## " section per assignment and its due date.
- Restart the application after editing these files.

## Project Structure

app.py - Gradio chat interface

chatbot.py - Retrieval, system prompt, and language model

cumberlands_info.md - University of the Cumberlands offices and links

course_schedules/ - Assignment due dates for added courses

requirements.txt - Python dependencies

## Limitations

- Answers are general guidance. Students should confirm important
  information with the university.
- The chatbot only knows the information in the knowledge files.
  It cannot access student accounts, grades, or live university
  systems.
- Due dates are only available for the courses that were added.
- A small language model can still give an incorrect or incomplete
  answer. The fact-check catches invented links, contacts, and
  dates, but not general wording mistakes (for example, made-up
  website steps or placeholder text).
- Answers take roughly 6-20 seconds on a laptop CPU.
- On Hugging Face Spaces, ZeroGPU has a daily GPU quota per visitor
  (about 2 minutes without a Hugging Face login, 5 minutes with a
  free account).
- A free Space goes to sleep when unused and takes about a minute
  to start again.

## Version History

- Version 1: Flask web app with TF-IDF question matching against
  predefined answers (no language model).
- Version 2 (current): Gradio app with an open-source language model
  and retrieval over University of the Cumberlands information,
  following the instructor's guidance to use free resources
  instead of paid APIs.

## Academic Use and Credits

This project was developed for the University of the Cumberlands
Artificial Intelligence for Human-Computer Interaction course
(MSAI 631). Reused code, libraries, and AI assistance are
credited in CREDITS.md.
