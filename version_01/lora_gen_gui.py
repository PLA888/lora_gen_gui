import os
import json
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from docx import Document
import pandas as pd
import PyPDF2
from openai import OpenAI
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import webbrowser

# 全局变量用于存储 API 配置
api_key_var = None
base_url_var = None
model_var = None
max_tokens_var = None


# 去除 Markdown 标记，提取 JSON 内容
def clean_markdown_content(raw_content):
    cleaned_content = re.sub(r'```json\s*|\s*```', '', raw_content, flags=re.DOTALL).strip()
    cleaned_content = re.sub(r'```\s*|\s*```', '', cleaned_content, flags=re.DOTALL).strip()
    return cleaned_content


# 使用 AI 生成问题和答案
def generate_qa_from_text(text):
    client = OpenAI(
        api_key=api_key_var.get(),
        base_url=base_url_var.get(),
    )

    prompt = f"""
    根据以下文本生成问题和对应的答案，要求问题简洁清晰，答案准确且基于文本内容，其中你一定要根据内容的长短判断生成多少个问题，如果我的内容很长很长，那么你就要全面化和细致化，
    就要生成很多个问题和答案，把所有内容都要用问题和答案概览进去，反之内容越短，根据语义，生成的问题和答案就越少。在生成答案的时候，一定要根据内容的上下文总结出最准确的答案。
    一定要遵从我约定好的输出格式,不要出现多余的文字，这一点非常重要,直接给我返回JSON，这是我的文本内容：{text}：
    输出格式：
    {{
      "question": "问题",
      "answer": "答案"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=model_var.get(),
            messages=[
                {"role": "system",
                 "content": "你是一个智能助手，擅长从文本中提取信息并生成问题和答案。一定要遵从约定好的输出格式,不要出现多余的文字，直接输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(max_tokens_var.get()),
            temperature=0.7
        )
        raw_content = response.choices[0].message.content
        cleaned_content = clean_markdown_content(raw_content)
        try:
            result = json.loads(cleaned_content)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "question" in result and "answer" in result:
                return [result]
            else:
                return [{"question": "解析失败", "answer": "API 返回格式不符合预期"}]
        except json.JSONDecodeError as e:
            return [{"question": "解析失败",
                     "answer": f"API 返回非 JSON 格式或解析错误: {str(e)} - 原始内容: {cleaned_content}"}]
    except Exception as e:
        return [{"question": "生成失败", "answer": f"API 调用出错: {str(e)}"}]


# 保存到 result.json 并实时更新显示
def save_to_json(qa_pairs, json_widget, filename="result.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.extend(qa_pairs)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    json_widget.delete(1.0, tk.END)
    json_widget.insert(tk.END, json.dumps(existing_data, ensure_ascii=False, indent=4))
    json_widget.see(tk.END)


# 改进的分段函数
def split_text_into_chunks(text, max_chunk_size=2000):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if not para.strip():
            continue
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

        while len(current_chunk) > max_chunk_size:
            split_point = current_chunk.rfind("\n", 0, max_chunk_size)
            if split_point == -1:
                split_point = max_chunk_size
            chunks.append(current_chunk[:split_point].strip())
            current_chunk = current_chunk[split_point:].strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# 处理单个段落
def process_paragraph(para, json_widget, progress_var, total_paragraphs, processed_count):
    qa_pairs = generate_qa_from_text(para)
    save_to_json(qa_pairs, json_widget)
    processed_count[0] += 1
    progress_var.set(f"进度: {processed_count[0]}/{total_paragraphs}")


# 文件读取和处理函数
def read_and_process_file(file_path, text_widget, json_widget, progress_var):
    file_extension = os.path.splitext(file_path)[1].lower()
    text = ""

    text_widget.delete(1.0, tk.END)

    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text_widget.insert(tk.END, f"TXT 文件内容 ({file_path}):\n{text}\n\n")
        elif file_extension == '.docx':
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            text_widget.insert(tk.END, f"Word 文件内容 ({file_path}):\n{text}\n\n")
        elif file_extension == '.xlsx':
            df = pd.read_excel(file_path)
            text = df.to_string()
            text_widget.insert(tk.END, f"Excel 文件内容 ({file_path}):\n{text}\n\n")
        elif file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            text_widget.insert(tk.END, f"PDF 文件内容 ({file_path}):\n{text}\n\n")
        else:
            text_widget.insert(tk.END, f"不支持的文件格式：{file_extension} ({file_path})\n\n")
            return

        chunks = split_text_into_chunks(text, max_chunk_size=1000)
        total_paragraphs = len(chunks)
        processed_count = [0]

        if total_paragraphs == 0:
            text_widget.insert(tk.END, "文件内容为空，无段落可处理。\n\n")
            return

        progress_var.set(f"进度: 0/{total_paragraphs}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(
                lambda para: process_paragraph(para, json_widget, progress_var, total_paragraphs, processed_count),
                chunks)

    except Exception as e:
        text_widget.insert(tk.END, f"处理文件 {file_path} 时出错：{str(e)}\n\n")


# 处理文件上传
def process_files(files, text_widget, json_widget, progress_var):
    if not api_key_var.get() or not base_url_var.get() or not model_var.get() or not max_tokens_var.get():
        text_widget.insert(tk.END, "请先填写 API Key、Base URL、Model 和 Max Tokens\n\n")
        return
    try:
        int(max_tokens_var.get())
    except ValueError:
        text_widget.insert(tk.END, "Max Tokens 必须为整数\n\n")
        return
    for file in files:
        Thread(target=read_and_process_file, args=(file, text_widget, json_widget, progress_var)).start()


# JSON 转换为 JSONL
def convert_json_to_jsonl(input_json_path, output_jsonl_path):
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for item in input_data:
                question = item.get("question", "").strip()
                answer = item.get("answer", "").strip()

                if not question or not answer:
                    continue

                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                json_obj = {"messages": messages}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"转换出错: {str(e)}")
        return False


# 导出 JSONL 文件
def export_jsonl(jsonl_widget):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jsonl",
        filetypes=[("JSONL 文件", "*.jsonl"), ("所有文件", "*.*")],
        title="导出 JSONL 文件"
    )
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(jsonl_widget.get(1.0, tk.END))
        jsonl_widget.insert(tk.END, f"\n已导出到: {file_path}")


# 刷新 result.json 内容
def refresh_json(json_widget):
    json_widget.delete(1.0, tk.END)
    if os.path.exists("result.json"):
        with open("result.json", "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                json_widget.insert(tk.END, json.dumps(data, ensure_ascii=False, indent=4))
            except json.JSONDecodeError:
                json_widget.insert(tk.END, "result.json 文件内容无效")
    else:
        json_widget.insert(tk.END, "result.json 文件不存在")


# 清除 result.json 内容
def clear_json(json_widget):
    if messagebox.askyesno("确认", "确定要清除 result.json 的所有内容吗？此操作不可撤销！"):
        json_widget.delete(1.0, tk.END)
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump([], f)  # 写入空数组
        json_widget.insert(tk.END, "[]")


# GUI 主程序
def create_gui():
    global root, api_key_var, base_url_var, model_var, max_tokens_var
    root = tk.Tk()
    root.geometry("1200x800")
    root.configure(bg="#f0f0f0")

    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 10), padding=5)
    style.configure("TLabel", font=("Helvetica", 12, "bold"), background="#f0f0f0")

    # 添加标题和可点击链接
    title_frame = ttk.Frame(root)
    title_frame.pack(fill=tk.X, pady=5)

    title_label = ttk.Label(title_frame, text="文件处理与 QA 生成工具 - ")
    title_label.pack(side=tk.LEFT)

    link_label = ttk.Label(title_frame, text="https://github.com/xliking/lora_gen_gui",
                          foreground="blue", cursor="hand2")
    link_label.pack(side=tk.LEFT)
    link_label.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/xliking/lora_gen_gui"))

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 添加 API 配置区域
    config_frame = ttk.LabelFrame(main_frame, text="API 配置", padding=10)
    config_frame.pack(fill=tk.X, pady=5)

    ttk.Label(config_frame, text="API Key:").pack(side=tk.LEFT, padx=5)
    api_key_var = tk.StringVar(value="xai-B7ZcEOD")
    api_key_entry = ttk.Entry(config_frame, textvariable=api_key_var, width=30)
    api_key_entry.pack(side=tk.LEFT, padx=5)

    ttk.Label(config_frame, text="Base URL:").pack(side=tk.LEFT, padx=5)
    base_url_var = tk.StringVar(value="https://api.x.ai/v1")
    base_url_entry = ttk.Entry(config_frame, textvariable=base_url_var, width=30)
    base_url_entry.pack(side=tk.LEFT, padx=5)

    ttk.Label(config_frame, text="Model:").pack(side=tk.LEFT, padx=5)
    model_var = tk.StringVar(value="grok-2-latest")
    model_entry = ttk.Entry(config_frame, textvariable=model_var, width=20)
    model_entry.pack(side=tk.LEFT, padx=5)

    ttk.Label(config_frame, text="Max Tokens:").pack(side=tk.LEFT, padx=5)
    max_tokens_var = tk.StringVar(value="8000")
    max_tokens_entry = ttk.Entry(config_frame, textvariable=max_tokens_var, width=10)
    max_tokens_entry.pack(side=tk.LEFT, padx=5)

    # 左侧框架
    left_frame = ttk.LabelFrame(main_frame, text="文件内容", padding=10)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    button_frame_left = ttk.Frame(left_frame)
    button_frame_left.pack(fill=tk.X, pady=5)

    upload_button = ttk.Button(button_frame_left, text="上传文件",
                               command=lambda: upload_files(text_widget, json_widget, progress_var))
    upload_button.pack(side=tk.LEFT, padx=5)

    clear_button = ttk.Button(button_frame_left, text="清空内容", command=lambda: text_widget.delete(1.0, tk.END))
    clear_button.pack(side=tk.LEFT, padx=5)

    text_widget = tk.Text(left_frame, height=30, width=40, font=("Helvetica", 10), wrap=tk.WORD)
    text_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=text_scroll.set)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.pack(fill=tk.BOTH, expand=True)

    progress_var = tk.StringVar(value="进度: 0/0")
    progress_label = ttk.Label(left_frame, textvariable=progress_var, font=("Helvetica", 10))
    progress_label.pack(pady=5)

    # 中间框架 (result.json)
    middle_frame = ttk.LabelFrame(main_frame, text="result.json 内容", padding=10)
    middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    button_frame_middle = ttk.Frame(middle_frame)
    button_frame_middle.pack(fill=tk.X, pady=5)

    refresh_button = ttk.Button(button_frame_middle, text="刷新", command=lambda: refresh_json(json_widget))
    refresh_button.pack(side=tk.LEFT, padx=5)

    clear_json_button = ttk.Button(button_frame_middle, text="清除", command=lambda: clear_json(json_widget))
    clear_json_button.pack(side=tk.LEFT, padx=5)

    json_widget = tk.Text(middle_frame, height=30, width=40, font=("Helvetica", 10), wrap=tk.WORD)
    json_scroll = ttk.Scrollbar(middle_frame, orient=tk.VERTICAL, command=json_widget.yview)
    json_widget.configure(yscrollcommand=json_scroll.set)
    json_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    json_widget.pack(fill=tk.BOTH, expand=True)

    if os.path.exists("result.json"):
        with open("result.json", "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                json_widget.insert(tk.END, json.dumps(data, ensure_ascii=False, indent=4))
            except json.JSONDecodeError:
                json_widget.insert(tk.END, "result.json 文件暂无内容")

    # 右侧框架 (jsonl 内容)
    right_frame = ttk.LabelFrame(main_frame, text="output.jsonl 内容", padding=10)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    button_frame_right = ttk.Frame(right_frame)
    button_frame_right.pack(fill=tk.X, pady=5)

    convert_button = ttk.Button(button_frame_right, text="转换为 JSONL",
                                command=lambda: convert_and_display(json_widget, jsonl_widget))
    convert_button.pack(side=tk.LEFT, padx=5)

    clear_jsonl_button = ttk.Button(button_frame_right, text="清空",
                                    command=lambda: jsonl_widget.delete(1.0, tk.END))
    clear_jsonl_button.pack(side=tk.LEFT, padx=5)

    export_button = ttk.Button(button_frame_right, text="导出",
                               command=lambda: export_jsonl(jsonl_widget))
    export_button.pack(side=tk.LEFT, padx=5)

    jsonl_widget = tk.Text(right_frame, height=30, width=40, font=("Helvetica", 10), wrap=tk.WORD)
    jsonl_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=jsonl_widget.yview)
    jsonl_widget.configure(yscrollcommand=jsonl_scroll.set)
    jsonl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    jsonl_widget.pack(fill=tk.BOTH, expand=True)

    def upload_files(text_widget, json_widget, progress_var):
        files = filedialog.askopenfilenames(
            title="选择文件",
            filetypes=[("支持的文件", "*.txt *.docx *.xlsx *.pdf"), ("所有文件", "*.*")]
        )
        if files:
            process_files(files, text_widget, json_widget, progress_var)

    def convert_and_display(json_widget, jsonl_widget):
        input_file = "result.json"
        output_file = "output.jsonl"
        if os.path.exists(input_file):
            if convert_json_to_jsonl(input_file, output_file):
                jsonl_widget.delete(1.0, tk.END)
                with open(output_file, 'r', encoding='utf-8') as f:
                    jsonl_widget.insert(tk.END, f.read())
                jsonl_widget.see(tk.END)
            else:
                jsonl_widget.delete(1.0, tk.END)
                jsonl_widget.insert(tk.END, "转换失败，请检查 result.json 文件")
        else:
            jsonl_widget.delete(1.0, tk.END)
            jsonl_widget.insert(tk.END, "result.json 文件不存在")

    root.mainloop()


if __name__ == "__main__":
    create_gui()