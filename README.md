It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts. This project uses venv.

For Linux:

# Create the virtual environment
```bash
python3 -m venv env-zeddg
```
# Activate it
```bash
source env-zeddg/bin/activate
```
After activation, your terminal prompt should be prefixed with (venv).

# Install PyTorch
To ensure hardware compatibility (especially for NVIDIA GPUs with CUDA), you must install PyTorch from its official source before installing other dependencies.

Go to the Official PyTorch 'Get Started' Page.

Use the interactive tool to select your OS, package manager (pip), and compute platform (CPU or your specific CUDA version).


[📄 Seecoder checkpoint](https://huggingface.co/shi-labs/prompt-free-diffusion/tree/main/pretrained/pfd/seecoder) | 

---

## Installation

Follow these steps to set up a local development environment.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
cd your-repository
```

```bash
# EXAMPLE ONLY: For CUDA 12.1. Do NOT copy this command.
# Generate your own command from the official PyTorch website.
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```
4. Install Project Dependencies
Once PyTorch is correctly installed, install the remaining dependencies from the requirements.txt file.

```bash

pip install -r requirements.txt
```

5. Install xformers (Optional)
For significantly improved performance and memory efficiency on compatible GPUs, install xformers.

Consult the official xformers GitHub [repository](https://github.com/facebookresearch/xformers) for advanced installation methods, such as building from source.
