from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_groq import ChatGroq  # ✅ Use Groq model here

import os
import streamlit as st

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

st.set_page_config(page_title="Smart Utech Chatbot 🤖", layout="wide")

st.title("🎓 Smart Utech Chatbot")
st.markdown("Ask me anything about UTECH! 💬")

# Step 1: Load static Utech info
@st.cache_resource
def setup_qa_chain():
    utech_data = [
        """
        UTech Digital Education is a UK-registered platform dedicated to empowering children aged 7–17 with digital skills, fostering independence and innovation.

        🏫 Onsite Campuses:
        - F-90 Block F NORTH NAZIMABAD, Karachi: Game Design, Python Level 1, Freelancing (Morning/Afternoon)
        - NED University Campus: Web Development, Digital Marketing (Evening slots)
        - MAJU Campus (Block 6 PECHS, Karachi): AI, Robotics, Python, Game Development

        💻 Online Courses:
        - Python Coding & Machine Learning (Basic to Advanced)
        - Graphics Designing (Adobe Illustrator & Photoshop)
        - AI and Robotics Adventures
        - 2D & 3D Game Development (Buildbox, Unity)
        - Cartoon Animation & Voice Over
        - Machine Learning with Python
        - Cloud Computing (AWS)
        - Android & iOS App Development
        - Graphics Designing & Freelancing

        🎓 Course Levels:
        - Beginner
        - Advanced
        - For Youngsters (ages 7–17)

        📞 Contact Numbers:
        - F-90 Campus: +92 321 UTECH KHI
        - NED Campus: +92 321 UTECH ONSITE
        - Online Courses: +92 321 UTECH ONLINE

        Founder:
        - Unais Ali

        🌐 Website: https://www.utech-edu.com
        📧 Email: mail@utech.edu.com
        📍 UK Office: 83 Bridge Street, Kington, United Kingdom, HR5 3DJ

        UTech offers both online and onsite classes, providing a flexible learning environment tailored to the needs of young learners. Their curriculum emphasizes practical skills, creativity, and problem-solving, preparing students for the technological challenges of the future.
        """
    ]

    # FAQ questions and answers
    faq_pairs = [
        ("What is UTECH?", "UTECH is a UK-registered digital education platform that empowers children aged 7–17 with digital skills."),
        ("Who is the founder of UTECH?", "The founder of UTECH is Unais Ali."),
        ("What does UTECH offer?", "UTECH offers online and onsite classes in tech fields like Python, AI, Robotics, Game Development, and more."),
        ("What is the age range for UTECH students?", "UTECH serves students aged 7–17."),
        ("Where is the UK office of UTECH?", "UTECH's UK office is at 83 Bridge Street, Kington, United Kingdom, HR5 3DJ."),
        ("What courses are available at the F-90 campus?", "Game Design, Python Level 1, and Freelancing."),
        ("What can I study at NED Campus?", "Web Development and Digital Marketing."),
        ("What courses are offered at the MAJU campus?", "AI, Robotics, Python, and Game Development."),
        ("Are the onsite classes in the morning or evening?", "F-90 has morning and afternoon; NED offers evening."),
        ("How do I register for onsite classes?", "Contact the campus directly or visit utech-edu.com."),
        ("What online courses does UTECH offer?", "Python Coding, Graphics Design, AI & Robotics, Game Development, and more."),
        ("Is Flutter available online?", "Yes, Flutter app development is available as an online course."),
        ("Can I learn AI or robotics online?", "Yes, UTECH offers AI and Robotics Adventures online."),
        ("Are online classes for beginners or advanced students?", "Both! UTECH has beginner and advanced online tracks."),
        ("How do I enroll in online classes?", "Visit utech-edu.com or call +92 321 UTECH ONLINE."),
        ("Are these courses for beginners?", "Yes, UTECH offers beginner-level courses and advanced programs."),
        ("Can kids join UTECH courses?", "Absolutely! UTECH is for students aged 7 to 17."),
        ("What’s the age limit to join?", "Students from ages 7 to 17 can join."),
        ("Do you have advanced level programs?", "Yes, UTECH provides advanced level courses."),
        ("Where is the F-90 campus located?", "Block F, Karachi."),
        ("What is the contact number for the NED campus?", "Call +92 321 UTECH NED."),
        ("How can I contact UTECH for online courses?", "Call +92 321 UTECH ONLINE or email mail@utech.edu.com."),
        ("What’s the email address for UTECH?", "mail@utech.edu.com."),
        ("Who teaches at UTECH?", "Courses are led by experienced instructors and mentors."),
        ("Who is on the UTECH team?", "The team includes Unais Ali and expert educators."),
        ("Can I get help from a mentor?", "Yes, UTECH provides mentorship with each course."),
        ("Will these courses help me freelance?", "Yes, especially Freelancing, Graphics Design, and Python are great for freelancing."),
        ("Can I become a game developer through UTECH?", "Yes, UTECH offers 2D/3D Game Development using Unity and Buildbox."),
        ("Is Python taught for real-world use?", "Yes, Python is taught with real-world use cases."),
        ("What is cartoon animation with voice-over?", "It's a course where students learn animation and voice-over techniques."),
        ("What platforms are used in game development?", "UTECH uses Unity and Buildbox."),
        ("Do you teach using Unity or Buildbox?", "Yes, both Unity and Buildbox are part of game dev courses."),
        ("What tools do you use in AI and robotics?", "UTECH uses educational kits and software to teach AI and robotics."),
        ("Any group for queries and help desk?", "https://chat.whatsapp.com/Bj4z0v9mEuA6BnG8IdxYu7"),
    ]

    # Combine static UTECH data with FAQs into documents
    docs = [Document(page_content=text) for text in utech_data]
    faq_docs = [Document(page_content=f"Q: {q}\nA: {a}") for q, a in faq_pairs]
    docs += faq_docs  # Add FAQ content to the vector store

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    # Update ChatGroq initialization
    llm = ChatGroq(
    temperature=0.3,
    model_name='llama3-70b-8192',  # Ensure this is provided if it's required
    api_key='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'  # Also ensure API key is included if required
)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = setup_qa_chain()

# Step 2: Smart Guided Flow
st.subheader("👇 Start your guided journey")

option = st.selectbox("What type of class are you looking for?", ["Online", "Onsite"])
campus = None
if option == "Onsite":
    campus = st.selectbox("Select Campus 🏫", ["F-90 Block F NORTH NAZIMABAD", "NED University"])

courses = []
if option == "Online":
    courses = ["Python Coding & Machine Learning (Basic to Advanced)", "Graphics Designing (Adobe Illustrator & Photoshop)", "AI and Robotics Adventures","2D & 3D Game Development (Buildbox, Unity)","Cartoon Animation & Voice Over"]
elif campus == "F-90 Block F NORTH NAZIMABAD":
    courses = ["Game Design", "Python Level 1", "Freelancing","ITC skills","Website design","Creative & Copy writing","AI Robotics & Adventure","Grapics Designing","Digital content creation"]
elif campus == "NED University":
    courses = ["AI Robotics & Adventure", "Cartoon Animation & voice over","ITC Skills","Digital Marketing & Freelancing","Python Data Science & Machine Learning","Advance website Developement","Python coding 1","Robotics","3D Game Developement","E-commerce website","Generative AI unsing Python","Machine Learning with Python","Cloud Computing (AWS)","Android & iOS App Development","Graphics Designing & Freelancing"]


course = st.selectbox("Select Course 📚", courses)
if course:
    st.success(f"You selected: {option} > {campus if campus else 'Online'} > {course}")
    contact_number = "📞 Contact: +92 321 UTECH ONLINE" if option == "Online" else "📞 Contact: +92 321 UTECH ONSITE"
    st.info(contact_number)

# Step 3: User Questions & Answers
user_query = st.text_input("Ask your question 🔍")

if user_query:
    result = qa_chain.run(user_query)
    st.write(result)
