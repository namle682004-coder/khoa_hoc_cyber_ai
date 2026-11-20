import gradio as gr

# =============================
# CORE LOGIC
# =============================

def grammar_fix(document, instruction, history):
    """
    document: nội dung hiện tại
    instruction: yêu cầu người dùng
    history: chat history
    """

    if not instruction.strip():
        return history, document, ""

    prompt = f"""
Bạn là công cụ sửa ngữ pháp chuyên nghiệp.

Văn bản:
{document}

Yêu cầu:
{instruction}

Hãy trả về phiên bản đã chỉnh sửa đầy đủ.
"""

    # 🔥 DEMO AI (mock) – thay bằng LLM thật sau
    fixed_text = document.strip()

    if "grammar" in instruction.lower() or "ngữ pháp" in instruction.lower():
        fixed_text = document.replace(" i ", " I ")

    if "formal" in instruction.lower():
        fixed_text = "This is a more formal version:\n\n" + document

    if not fixed_text:
        fixed_text = document

    history.append((instruction, "✍️ Đã cập nhật văn bản trong editor."))

    return history, fixed_text, ""


# =============================
# CSS (Word-like Layout)
# =============================

css = """
body {
    background:#0f1117;
}

.gradio-container {
    max-width:100% !important;
}

#menu-bar {
    background:#ffffff;
    padding:8px 20px;
    border-bottom:1px solid #ddd;
    font-weight:600;
}

#ribbon {
    background:#f5f5f5;
    padding:8px 20px;
    border-bottom:1px solid #ddd;
}

#editor textarea {
    font-size:16px;
    line-height:1.6;
    background:white;
    min-height:65vh;
}

#chatbox {
    height:180px;
    overflow-y:auto;
}

#prompt-row {
    position:sticky;
    bottom:0;
    background:#0f1117;
    padding:10px;
    border-top:1px solid #333;
}
"""

# =============================
# UI
# =============================

with gr.Blocks() as demo:

    # ---- MENU BAR ----
    gr.HTML("""
    <div id="menu-bar">
        File &nbsp; Home &nbsp; Insert &nbsp; Layout &nbsp; Review &nbsp; Spelling & Grammar
    </div>
    """)

    # ---- RIBBON ----
    gr.HTML("""
    <div id="ribbon">
        Language:
        <select>
            <option>English</option>
            <option>Vietnamese</option>
        </select>
        &nbsp;&nbsp; Font:
        <select>
            <option>Calibri</option>
            <option>Arial</option>
            <option>Times</option>
        </select>
    </div>
    """)

    # ---- DOCUMENT ----
    document = gr.Textbox(
        label="Document",
        elem_id="editor",
        placeholder="Dán hoặc viết văn bản của bạn ở đây...",
        lines=30
    )

    # ---- CHAT HISTORY ----
    chatbot = gr.Chatbot(label="AI Assistant", elem_id="chatbox")
    state = gr.State([])

    # ---- PROMPT BAR ----
    with gr.Row(elem_id="prompt-row"):
        instruction = gr.Textbox(
            placeholder="Nhập yêu cầu... ví dụ: sửa ngữ pháp, viết học thuật hơn, rút gọn 20%",
            show_label=False,
            scale=9
        )
        send = gr.Button("Apply", scale=1)

    # ---- EVENTS ----
    send.click(
        grammar_fix,
        inputs=[document, instruction, state],
        outputs=[chatbot, document, instruction]
    )

    instruction.submit(
        grammar_fix,
        inputs=[document, instruction, state],
        outputs=[chatbot, document, instruction]
    )

# =============================
# LAUNCH
# =============================

demo.launch(
    css=css,
    server_name="127.0.0.1",
    server_port=7865
)
