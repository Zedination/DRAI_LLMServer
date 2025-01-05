import multiprocessing

from llama_cpp import llama_cpp
from llama_cpp.server.app import create_app
import os
import uvicorn
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command-line arg for local llm server")

    # define argument
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    parser.add_argument("--model", type=str, required=True, help="The path to the model to use for generating completions.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Listen address")
    parser.add_argument("--n_gpu_layers", type=int, default=0, help="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.")
    parser.add_argument("--seed", type=int, default=llama_cpp.LLAMA_DEFAULT_SEED, help="Random seed. -1 for random.")
    parser.add_argument("--n_ctx", type=int, default=2048, help="The context size.")
    parser.add_argument("--n_threads", default=max(multiprocessing.cpu_count() // 2, 1), help="The number of threads to use. Use -1 for max cpu threads")

    args = parser.parse_args()

    for arg in vars(args):
        os.environ[arg.upper()] = getattr(args, arg)

    app = create_app()

    uvicorn.run(app, host=args.host, port=args.port)
