import tempfile
import re
import requests
from openai import OpenAI
import pandas as pd
import json
import traceback
import fitz  # PyMuPDF
import base64
from urllib.parse import urlparse
import mimetypes

from dotenv import load_dotenv
import os

load_dotenv()
metis_api_key = os.getenv("METIS_API_BASE")


class DivarContest:
    def __init__(self, api_token):
        self.api_token = api_token
        self.model = "gpt-4.1-mini"
        self.client = OpenAI(api_key=self.api_token, base_url="https://api.metisai.ir/openai/v1")
        self.workspace = tempfile.mkdtemp()

    def download_file_tool(self, url, destination=None):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Try to get filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)

            # If URL doesn't provide a filename, try to get it from headers
            if not filename or '.' not in filename:
                content_disposition = response.headers.get('content-disposition')
                if content_disposition:
                    # Try to extract filename from content-disposition
                    import re
                    match = re.search('filename="?([^";]+)"?', content_disposition)
                    if match:
                        filename = match.group(1)
                if not filename:
                    # Fallback to generic name with inferred extension
                    ext = mimetypes.guess_extension(response.headers.get('content-type', '').split(';')[0].strip())
                    filename = f"downloaded_file{ext or ''}"

            if destination is None:
                destination = os.path.join(self.workspace, filename)

            # Write to file
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ File downloaded to: {destination}")
            return destination

        except Exception as e:
            return f"❌ Error downloading file: {str(e)}"

    def summarize_dataframe(self, df):
        """Create a summary string with schema and sample rows."""
        shape = df.shape
        schema = ", ".join(f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items())
        sample_rows = df.head(15).to_dict(orient="records")

        return {
            "shape": shape,
            "schema": schema,
            "sample_rows": sample_rows
        }

    def safe_execute_code(self, code: str, df: pd.DataFrame):
        # print("Trying to run the generated code")
        local_vars = {"df": df}
        try:
            exec(code, {}, local_vars)
            # print("Finished executing the generated code")
            return str(local_vars.get("answer", "Error: 'answer' variable not set."))
        except Exception as e:
            return f"Execution Error: {str(e)}\n\n{traceback.format_exc()}"

    def answer_from_excel_tool(self, excel_file_url, question):
        file_path = self.download_file_tool(excel_file_url)

        try:
            # Try reading as Excel
            df = pd.read_excel(file_path)
        except Exception as e_excel:
            print("Failed to read as Excel:", e_excel)
            try:
                # Fallback: Try reading as CSV with utf-8, then latin1
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
            except Exception as e_csv:
                return f"Could not read the file as Excel or CSV:\nExcel error: {e_excel}\nCSV error: {e_csv}"

        df_summary = self.summarize_dataframe(df)

        query_prompt = f"""
            You are a data analyst assistant. The user has asked this question:
            \"\"\"{question}\"\"\"
            
            The dataset is loaded into a pandas DataFrame called `df`. To get familiar with the DataFrame, you are given the schema and some of the first rows of the df as the sample:
            
            DataFrame's Shape: {df_summary["shape"]}
            
            DataFrame's Schema: {df_summary["schema"]}
            
            DataFrame's Sample Rows: {df_summary["sample_rows"]}
            
            If the question was a general question about the DataFrame (such as the shape of the file or the number of columns) return the answer, but if answering the question required additional information (such as questions about the sales which require seeing the whole data), then do the following:
            
            Write a Python snippet (using pandas) that answers the user's question.
            Store the final numeric/string result in a variable named `answer`.
            Do not print anything. Just write the code.
        """

        code_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query_prompt}],
            max_tokens=500,
            temperature=0.1
        )
        generated_code = code_response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
        with open("generated_code.py", "w", encoding="utf-8") as f:
            f.write(generated_code)

        return self.safe_execute_code(generated_code, df)

    def audio_transcript_tool(self, audio_file_url):
        file_name = self.download_file_tool(audio_file_url)

        try:
            audio_file = open(file_name, "rb")

        except FileNotFoundError:
            filename = os.path.basename(file_name)
            for root, _, files in os.walk(self.workspace):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    try:
                        audio_file = open(found_path, "rb")
                    except:
                        return f"Error: File {file_name} not found"
        except Exception as e:
            return f"Error reading file: {str(e)}"

        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="json",
            temperature=0
        )
        print(transcription)
        return transcription.text

    def read_pdf_tool(self, pdf_file_url):
        """
        Downloads and extracts full text content from a PDF file.
        Returns the extracted text as a single string.
        """
        file_name = self.download_file_tool(pdf_file_url)
        try:
            with fitz.open(file_name) as doc:
                pdf_text = ""
                for page in doc:
                    pdf_text += page.get_text()
            return pdf_text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def get_image_data_tool(self, image_file_url):
        try:
            response = requests.get(image_file_url, timeout=30)
            response.raise_for_status()

            # Convert image to base64
            image_data = base64.b64encode(response.content).decode("utf-8")

            # Use OpenAI client from the agent's context
            # return {"image_base64": image_data, "url": image_file_url}
            return image_data
        except Exception as e:
            return f"Error extracting image: {str(e)}"

    def capture_the_flag(self, question):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "answer_from_excel_tool",
                    "description": "Downloads an excel file from a URL of the internet, then extracts the needed information from the excel file and returns the final answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "excel_file_url": {
                                "type": "string",
                                "description": "The URL of the excel file to fetch."
                            },
                            "question": {
                                "type": "string",
                                "description": "the question that should be answered using the data of the excel."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "audio_transcript_tool",
                    "description": "Downloads an audio file from a URL of the internet, then returns the transcript of the audio.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "audio_file_url": {
                                "type": "string",
                                "description": "The URL of the audio file to fetch."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_pdf_tool",
                    "description": "Downloads a PDF file from a URL of the internet and returns the data in the PDF file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pdf_file_url": {
                                "type": "string",
                                "description": "The URL of the PDF file to fetch."
                            }
                        }
                    }
                }

            },
            {
                "type": "function",
                "function": {
                    "name": "get_image_data_tool",
                    "description": "Downloads an Image a URL of the internet and returns the data in the image file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_file_url": {
                                "type": "string",
                                "description": "The URL of the image file to fetch."
                            }
                        }
                    }
                }

            }
        ]

        function_map = {
            "answer_from_excel_tool": self.answer_from_excel_tool,
            "audio_transcript_tool": self.audio_transcript_tool,
            "read_pdf_tool": self.read_pdf_tool,
            "get_image_data_tool":self.get_image_data_tool
        }

        system_prompt = """
        You are a helpful AI agent participating in a Capture the Flag (CTF) competition. You are provided with access to a set of tools (functions) to help you extract the required information from a file provided via URL.

        Your job is to:
        1. Understand the user’s question.
        2. Detect if the question references a file (Excel, audio, PDF, or image).
        3. Call the appropriate tool to extract data from that file.
        4. Post-process the tool's result (if needed) and return the **final answer only**, exactly in the expected format.

        📦 Tools Available:
        - `answer_from_excel_tool`: Use this if the question is about data in an Excel file (.xlsx or .csv).
        - `audio_transcript_tool`: Use this if the question involves spoken content in an audio file.
        - `read_pdf_tool`: Use this if the question asks for something mentioned inside a PDF document.
        - `get_image_data_tool`: Use this to analyze the content of an image, such as counting objects or extracting visual information.

        🧠 Behaviors:
        - Always use the correct tool when a file is present in the question.
        - Only return the final answer as plain text or in JSON if explicitly requested.
        - **Do explain your reasoning**, do say what you’re doing — return the answer in the required format.
        - If the result needs formatting (e.g., "USD with two decimal places", or lowercase text with no period), follow it strictly.
        - If the input contains no file, answer it directly if possible.
        - If you are asked to count the number of shapes in an image, analyze the content of an image and count the number of primary geometric shapes (circles, squares, triangles, etc.), ignoring text, shadows, background patterns. Only solid, visible, distinct shapes should be counted.

        🎯 Examples:
        Input:
        "The Excel file in this url: https://... contains sales. What were the total food sales?"
        → Use `answer_from_excel_tool` and return something like: `"89724.00"`

        Input:
        "What did the person say in this audio: https://...?"
        → Use `audio_transcript_tool` and return something like: `"may 30th, 2023"`

        Input:
        "How many accounts were blocked based on this PDF: https://...?"
        → Use `read_pdf_tool` and return something like: `"301000"`

        Input:
        "How many shapes are in this image: https://...?"
        → Use `get_image_data_tool` and return something like: `{"count": 6}`

        Be accurate, direct, and always match the output format exactly as required.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        max_turns = 3
        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0
            )

            print(response)
            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            if (
                hasattr(assistant_message, "tool_calls")
                and assistant_message.tool_calls
            ):
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    print(function_name)
                    function_args = json.loads(tool_call.function.arguments)

                    # Execute the function
                    tool_fn = function_map[function_name]
                    result = tool_fn(**function_args)
                    print(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result,
                        }
                    )
            else:
                return assistant_message.content.strip()


divar_contest = DivarContest(metis_api_key)

final_response = divar_contest.capture_the_flag("The Excel file in this url: https://3003.filemail.com/api/file/get?filekey=aO-fXPlI-dr2EXRNRkbCpEUFrPTI4N5WeH4OoMXuiQPBlZ1vjqNnX3aijCs7-DoXKzdPr1ZQ&pk_vid=01e1a289c298d03917536932675a7ae9 contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.")
# final_response = divar_contest.capture_the_flag("`Hi, one person said something in the sound file in this url: https://2011.filemail.com/api/file/get?filekey=8hpJun2PRuxe6i38x2MTHB2T2fSkKx6Qb5l6EmEVkxr7t3aflKuA_KL5vmfeot4UkkzNl_n_RGUXUvWKtEsqoSHEFGsWQ1LwtkfacQ&pk_vid=01e1a289c298d03917537172995a7ae9 . what she said?`")
# final_response = divar_contest.capture_the_flag("read the pdf in this url: https://download1339.mediafire.com/z55ry5d53jlgIp_1uHtDiLAv8mr5oVurBk7eQJFgPwRWblusE976uPC88sIjtzgb1tHV1wK8SHRAxQvnbHdkqB3OwHopyjtwEFaeTcWJt6z7mdccHGDqdmkN2txEJe0W_q7vjjYNN_Pb6eESVl2pIazP2H_xx5AvKdB49nrLCnDavA/9qy09xl6wqxoo9k/pdf+test.pdf and answer how many account were blocked based on pdf data")
# final_response = divar_contest.capture_the_flag("`what shapes are in the image in this url: https://download1085.mediafire.com/ralp5rlo6txgAgwzqUW-0VZGMVeS3XKX4aAPfvQYywUIKM3dwT3GWWAWU6NqI11tipfViFAvqWlbwqmMHs5SAyqiYuR_mT36haBhzjrUZ5f4yhin4y-B6ZN1ivGbzkTu0MoKGs5wjIyXTJsXr0PVr0waC5C45uxig840l1j986UJRA/lj0joelaggh592c/ChatGPT+Image+Jul+28%2C+2025%2C+07_45_31+PM.jpg ? Please provide the answer in a JSON format with the keys 'count'.`")

# final_response = divar_contest.get_image_data_tool("https://download1085.mediafire.com/ralp5rlo6txgAgwzqUW-0VZGMVeS3XKX4aAPfvQYywUIKM3dwT3GWWAWU6NqI11tipfViFAvqWlbwqmMHs5SAyqiYuR_mT36haBhzjrUZ5f4yhin4y-B6ZN1ivGbzkTu0MoKGs5wjIyXTJsXr0PVr0waC5C45uxig840l1j986UJRA/lj0joelaggh592c/ChatGPT+Image+Jul+28%2C+2025%2C+07_45_31+PM.jpg")
# final_response = divar_contest.read_pdf_tool("https://download1339.mediafire.com/z55ry5d53jlgIp_1uHtDiLAv8mr5oVurBk7eQJFgPwRWblusE976uPC88sIjtzgb1tHV1wK8SHRAxQvnbHdkqB3OwHopyjtwEFaeTcWJt6z7mdccHGDqdmkN2txEJe0W_q7vjjYNN_Pb6eESVl2pIazP2H_xx5AvKdB49nrLCnDavA/9qy09xl6wqxoo9k/pdf+test.pdf")
# final_response = divar_contest.audio_transcript_tool("https://2011.filemail.com/api/file/get?filekey=8hpJun2PRuxe6i38x2MTHB2T2fSkKx6Qb5l6EmEVkxr7t3aflKuA_KL5vmfeot4UkkzNl_n_RGUXUvWKtEsqoSHEFGsWQ1LwtkfacQ&pk_vid=01e1a289c298d03917537172995a7ae9")
# final_response = divar_contest.answer_from_excel_tool("https://3003.filemail.com/api/file/get?filekey=aO-fXPlI-dr2EXRNRkbCpEUFrPTI4N5WeH4OoMXuiQPBlZ1vjqNnX3aijCs7-DoXKzdPr1ZQ&pk_vid=01e1a289c298d03917536932675a7ae9", "What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places")
print(final_response)
