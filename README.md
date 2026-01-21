# Domain-Aware Evaluation

This repository contains code for domain-aware evaluation of Evo2.

## Prerequisites

- Python 3.12
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/samasalarian/domain-aware-evaluation.git
cd domain-aware-evaluation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the evaluation scripts from the `domain-aware-evaluation` directory:

### Evolution Analysis
```bash
python code/Conservation_A.py
```

### Generative Analysis
```bash
python code/Validity.py
```

### Clinical Analysis
```bash
python code/VEP.py
```

## Output

All generated results will be saved in the `outputs/` folder.
