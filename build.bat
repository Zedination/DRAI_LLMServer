pyinstaller --onefile --add-binary ".venv/Lib/site-packages/llama_cpp/lib/*.dll;llama_cpp" drai_llm_server.py