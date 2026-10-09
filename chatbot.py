
import os
import re
from pathlib import Path

# Hugging Face ZeroGPU support. It must be imported before torch, and it
# has no effect when the app runs on a laptop.
import spaces
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


# Free, open-source instruction model from Hugging Face (Apache 2.0).
# Change to "Qwen/Qwen2.5-1.5B-Instruct" for better-worded but slower answers.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILES = [BASE_DIR / "cumberlands_info.md"] + sorted(
    (BASE_DIR / "course_schedules").glob("*.md")
)

# Number of knowledge sections passed to the model for each question.
TOP_SECTIONS = 3

# Number of earlier chat messages the model sees.
HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are the University of the Cumberlands Student Support Chatbot,
a virtual assistant that helps University of the Cumberlands students with
routine academic and administrative questions.

You help with: assignment deadlines, academic expectations, academic
advising, course registration, financial aid and tuition, library services,
IT support and logins, university tools (course site, student portal,
email), and student support services.

Rules:
- Use only the University of the Cumberlands information provided below.
  Only share links, emails, and phone numbers that appear in it. Never
  invent dates, fees, names, links, contacts, or policies.
- Only give an assignment due date if that exact assignment and its date
  appear in the information below. Otherwise, tell the student to check
  their course page and syllabus.
- If the information below does not answer the question, say so and direct
  the student to the most relevant office listed below.
- Never ask for or accept passwords, student ID numbers, or other personal
  information. You cannot see student accounts or grades. If a student
  shares a password or student ID, remind them never to share these in a
  chat.
- If a question is unrelated to university student support, reply only:
  "I can only help with University of the Cumberlands student questions,
  such as deadlines, advising, registration, financial aid, the library,
  and IT support."
- Keep answers short (under 120 words), friendly, and in plain language."""

# Maximum length of a generated answer, in tokens.
MAX_NEW_TOKENS = 300

OFF_TOPIC_MESSAGE = (
    "I can only help with University of the Cumberlands student "
    "questions, such as deadlines, advising, registration, financial "
    "aid, the library, and IT support."
)

PRIVACY_WARNING = (
    "Please never share your password or student ID in a chat."
)

CANNOT_CONFIRM = (
    "I couldn't confirm that from official University of the "
    "Cumberlands information."
)

DEADLINE_HINT = (
    "For assignment due dates, please check your course page and "
    "syllabus on Blackboard (iLearn): https://ucumberlands.blackboard.com/"
)

DEFAULT_RESOURCE = (
    "the Current Students page: https://www.ucumberlands.edu/current-students"
)

MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)


def load_sections():
    """Split the knowledge files into sections, one per '## ' heading."""

    sections = []

    for path in KNOWLEDGE_FILES:
        text = path.read_text(encoding="utf-8")

        # The text before the first '## ' heading is the file's introduction.
        for part in text.split("\n## ")[1:]:
            sections.append("## " + part.strip())

    return sections


def extract_facts(text):
    """Find links, emails, phone numbers, dates, and dollar amounts."""

    facts = set()

    for url in re.findall(r"https?://[^\s)\]>\"']+", text):
        facts.add(url.rstrip("/.,;:").lower())

    for email in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text):
        facts.add(email.lower())

    for phone in re.findall(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text):
        facts.add(re.sub(r"\D", "", phone))

    for amount in re.findall(r"\$\d[\d,]*(?:\.\d+)?", text):
        facts.add(amount)

    return facts


def extract_dates(text):
    """Find dates such as 'October 11' or '10/11/2026'."""

    dates = {
        f"{month} {int(day)}"
        for month, day in re.findall(rf"({MONTHS})\s+(\d{{1,2}})", text)
    }
    dates.update(re.findall(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text))

    return dates


def find_course_codes(text):
    """Find course codes such as 'MSAI 631' and return them as 'MSAI631'."""

    return {
        f"{letters.upper()}{digits}"
        for letters, digits in re.findall(r"\b([A-Za-z]{2,5})\s?(\d{3})\b", text)
    }


def date_is_supported(date, question):
    """A date is allowed only if it comes from a schedule entry that the
    question is about (same course code and assignment name)."""

    question_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    question_codes = find_course_codes(question)

    for section in sections:
        if date not in extract_dates(section):
            continue

        heading = section.splitlines()[0]
        heading_codes = find_course_codes(heading)

        if question_codes and not question_codes & heading_codes:
            continue

        # Assignment name words, e.g. "results" and "report".
        name_words = set(re.findall(r"[a-z]+", heading.lower()))
        name_words -= {"msai", "group", "project", "course"}

        if name_words & question_words:
            return True

    return False


# Prepare the retrieval step (TF-IDF over the knowledge sections).
sections = load_sections()

# Every fact the chatbot is allowed to state.
known_facts = extract_facts("\n".join(sections))

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

section_vectors = vectorizer.fit_transform(sections)


# Load the language model once when the app starts.
# The first run downloads the model from Hugging Face.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")

# Use the GPU on a ZeroGPU Space (or a laptop with an NVIDIA GPU),
# otherwise the CPU.
if os.getenv("SPACES_ZERO_GPU") or torch.cuda.is_available():
    model.to("cuda")


def find_relevant_sections(question):
    """Return the knowledge sections most similar to the question."""

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, section_vectors)[0]

    best_indexes = similarities.argsort()[::-1][:TOP_SECTIONS]

    return [sections[i] for i in best_indexes if similarities[i] > 0]


@spaces.GPU(duration=60)
def run_model(inputs):
    """Generate an answer. On a ZeroGPU Space, a GPU is used only here."""

    inputs = inputs.to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    return output.cpu()


def safe_answer(question, relevant_sections):
    """Answer used when the model states a fact that cannot be verified."""

    resource = DEFAULT_RESOURCE

    # Point to the first relevant section that has an official link.
    for section in relevant_sections:
        link = re.search(r"^Link: (\S+)", section, re.MULTILINE)
        if link:
            name = section.splitlines()[0].removeprefix("## ")
            resource = f"{name}: {link.group(1)}"
            break

    answer = f"{CANNOT_CONFIRM} Please check {resource}"

    if re.search(r"\b(due|deadline|exam|assignment|homework)", question, re.I):
        answer += f"\n\n{DEADLINE_HINT}"

    return answer


def get_response(message, history):
    """Answer a student's question, adding a privacy warning when the
    student appears to share a password or student ID."""

    if not isinstance(message, str) or not message.strip():
        return "Please enter a question so I can help you."

    answer = generate_answer(message, history)

    if re.search(r"\b(password|passcode|student id|id number)\s*(is|:|=)",
                 message, re.I):
        answer = f"{PRIVACY_WARNING}\n\n{answer}"

    return answer


def generate_answer(message, history):
    """Answer a student's question using the retrieved information."""

    # Search with the previous question too, so follow-ups such as
    # "What is their phone number?" find the right office.
    previous_questions = [
        item["content"] for item in history
        if item.get("role") == "user" and isinstance(item.get("content"), str)
    ]
    search_text = " ".join(previous_questions[-1:] + [message])

    relevant_sections = find_relevant_sections(search_text)

    # No related university information: the question is off-topic.
    if not relevant_sections:
        return OFF_TOPIC_MESSAGE

    context = "\n\n".join(relevant_sections)

    messages = [{
        "role": "system",
        "content": (
            f"{SYSTEM_PROMPT}\n\n"
            f"University of the Cumberlands information:\n\n{context}"
        )
    }]

    # Include recent chat history so follow-up questions make sense.
    for item in history[-HISTORY_MESSAGES:]:
        if isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    output = run_model(inputs)

    # Decode only the newly generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # If the answer hit the length limit, cut it at the last full sentence.
    if len(new_tokens) >= MAX_NEW_TOKENS:
        cut = max(answer.rfind("\n"), answer.rfind(". "))
        if cut > 0:
            answer = answer[:cut + 1].strip()

    # Fact-check: every link, email, phone number, and amount must appear
    # in the knowledge files, and every date must belong to the assignment
    # the student asked about.
    unsupported_dates = [
        date for date in extract_dates(answer)
        if not date_is_supported(date, search_text)
    ]

    if not extract_facts(answer) <= known_facts or unsupported_dates:
        return safe_answer(message, relevant_sections)

    return answer
