
import gradio as gr

from chatbot import get_response


demo = gr.ChatInterface(
    fn=get_response,
    title="University of the Cumberlands Student Support Chatbot",
    description=(
        "Ask about deadlines, advising, registration, financial aid, "
        "library services, IT support, and student services. "
        "Student project for MSAI 631. Not an official University of "
        "the Cumberlands service; confirm important information with "
        "the university."
    ),
    examples=[
        "How do I contact financial aid?",
        "When is the MSAI 631 results report due?",
        "How do I reset my password?",
        "How do I access library databases?"
    ]
)


if __name__ == "__main__":
    demo.launch()
