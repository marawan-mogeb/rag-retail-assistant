"""
Streamlit frontend for the RAG-Powered Retail Assistant.

Chat-style interface: ask a question (optionally with a shelf image),
see a grounded answer with cited sources and, if an image was attached,
the YOLO detection summary.
"""
import streamlit as st

from api_client import API_BASE_URL, APIError, check_health, send_query

st.set_page_config(page_title="RAG Retail Assistant", page_icon="🛍️", layout="centered")

st.title("🛍️ RAG Retail Assistant")
st.caption(f"Backend: {API_BASE_URL}")

# --- Backend health check, shown once per session ---
if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = check_health()

if not st.session_state.backend_healthy:
    st.error(
        f"⚠️ Cannot reach the backend at `{API_BASE_URL}`. "
        "Make sure the FastAPI server is running (`uvicorn app.main:app --reload`), "
        "then refresh this page."
    )
    st.stop()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": ..., "sources": [...], "detection": {...}}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("image"):
            st.image(msg["image"], width=200)
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📚 Sources: " + ", ".join(msg["sources"]))
        if msg.get("detection"):
            d = msg["detection"]
            st.caption(
                f"🔍 Detected {d['num_products_detected']} products on shelf "
                f"(avg confidence: {d['avg_confidence']})"
            )

# --- Sidebar: optional image upload ---
with st.sidebar:
    st.header("Shelf Image (optional)")
    st.write(
        "Attach a shelf photo to have YOLO detect products on it. "
        "The detection result will be combined with your question."
    )
    uploaded_image = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", width=300)

# --- Chat input ---
question = st.chat_input("Ask about return policies, shipping, restocking...")

if question:
    image_bytes = uploaded_image.getvalue() if uploaded_image else None
    image_filename = uploaded_image.name if uploaded_image else "upload.jpg"

    # Show the user's message immediately
    with st.chat_message("user"):
        if uploaded_image:
            st.image(uploaded_image, width=200)
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "image": uploaded_image.getvalue() if uploaded_image else None,
    })

    # Get and show the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = send_query(question, image_bytes=image_bytes, image_filename=image_filename)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                detection = result.get("detection")

                st.markdown(answer)
                if sources:
                    st.caption("📚 Sources: " + ", ".join(sources))
                if detection:
                    st.caption(
                        f"🔍 Detected {detection['num_products_detected']} products on shelf "
                        f"(avg confidence: {detection['avg_confidence']})"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "detection": detection,
                })
            except APIError as e:
                error_msg = f"⚠️ {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
