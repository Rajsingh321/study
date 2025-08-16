from phi.agent import Agent
from phi.tools.youtube_tools import YouTubeTools
from phi.model.groq import Groq
from dotenv import load_dotenv
import streamlit as st
import google as genai
from groq import Groq as gr
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()


def generate_pdf(notes_text, title="AI Generated Notes"):
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Add title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # Add notes
    for paragraph in notes_text.split("\n"):
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), styles["Normal"]))
            story.append(Spacer(1, 8))

    pdf.build(story)
    buffer.seek(0)
    return buffer   # ✅ only return PDF file object

# ------------------ AI Function ------------------
def Notes_maker(summary):
    messages = [
        {
            "role": "system",
            "content": "You are an expert note-taking assistant for students."
        },
        {
            "role": "user",
            "content": f"""
Create detailed study notes from the following summary:
- Include structured topics and subtopics
- Include key points under each subtopic
- Add 2-5 likely exam questions at the end
- Use clear, concise, student-friendly language

Summary:
{summary}
""" 
        }
    ]
    client = gr(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages= messages,
        model="llama-3.3-70b-versatile",
        stream=False,
    )
    reply = chat_completion.choices[0].message.content.strip()
    
    return reply

# ------------------ Core Function ------------------
def youtube_video_summary(link,mode):
    
    agent = Agent(
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
        tools=[YouTubeTools()],
        show_tool_calls=True,
        description=(
            "You are a YouTube summarizer for students. "
            "Extract captions & metadata from the video, then create a clear, concise, and easy-to-read summary. "
            "Focus on key points, definitions, and important facts useful for exam preparation."
        )
    )
    output = agent.run(f"Create {mode.lower()} from the YouTube video this the link {link}. Include title, structured topics/subtopics,with short definition key points, and 2-5 exam questions at the end.")
    return output.content

# ------------------ Streamlit UI ------------------
def main():
    st.set_page_config(page_title="YouTube Study Summarizer")

    # Title & Description
    st.markdown("<h1 style='text-align:center;'>🎓 YouTube Study Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Turn long lectures into short, exam-ready notes</p>", unsafe_allow_html=True)

    # Input field
    link = st.text_input("📹 Paste YouTube video link here", placeholder="https://www.youtube.com/watch?v=...")
    mode = st.selectbox("Choose output type:", ["Quick Summary", "Detailed PDF Notes"])
    # Styling for button
    st.markdown("""
    <style>
        div.stButton > button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 20px;
            font-size: 16px;
            border-radius: 12px;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

    # Process video
    if st.button("📄 Generate Summary"):
        if link.strip() == "":
            st.error("⚠ Please enter a valid YouTube link.")
        else:
            with st.spinner("⏳ Summarizing the video for study notes..."):
                summary = youtube_video_summary(link,mode)
                if mode == "Detailed PDF Notes":
                    notes = Notes_maker(summary)
                    st.markdown("### 📄 Detailed Notes")
                    st.text_area("Notes", notes, height=300)
                    #st.markdown(notes)
                    pdf_file = generate_pdf(notes, title="AI Generated Notes")
                    st.download_button(
                    label="⬇ Download PDF",
                    data=pdf_file,
                    file_name="detailed_notes.pdf",
                    mime="application/pdf"
                    )
                else:
                    st.markdown("#📄 Quick Summary")
                    st.markdown(summary)

# ------------------ Run App ------------------
if __name__ == "__main__":
    st.title("NovaMind", anchor="top")
    main()
    