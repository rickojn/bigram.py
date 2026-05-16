# Bigram Language Model

A simple word-level bigram language model in Python. Feed it any text file and it will learn which words tend to follow which other words, then let you generate new text from a prompt.

---

## Requirements

- **Python 3.10 or newer** — no third-party packages needed, only the Python standard library.

---

## Step 1 — Install Python

If you already have Python 3.10+ installed, skip this step. To check, open a terminal and run:

```
python --version
```

or on some systems:

```
python3 --version
```

If you see `Python 3.10.x` or higher you are good to go. Otherwise:

- **Windows** — download the installer from [python.org/downloads](https://www.python.org/downloads/) and run it. On the first screen, tick **"Add Python to PATH"** before clicking Install.
- **macOS** — download the installer from [python.org/downloads](https://www.python.org/downloads/) and run it, or install via [Homebrew](https://brew.sh/) with `brew install python`.
- **Linux** — use your package manager, e.g. `sudo apt install python3` on Ubuntu/Debian.

---

## Step 2 — Download the files

### Option A — Download as a ZIP (no Git required)

1. Go to the repository page on GitHub.
2. Click the green **`<> Code`** button near the top right.
3. Click **Download ZIP**.
4. Once downloaded, right-click the ZIP file and choose **Extract All** (Windows) or double-click it (macOS/Linux) to unzip it.
5. Open the extracted folder — you should see `bigram_model.py` and `pride_and_prejudice_p1-5.txt` inside.

### Option B — Clone with Git (if you have Git installed)

```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

## Step 3 — Open a terminal in the project folder

- **Windows** — open the extracted folder in File Explorer, then click the address bar at the top, type `cmd`, and press Enter. A Command Prompt will open already pointing at the right folder.
- **macOS** — right-click the folder in Finder and choose **New Terminal at Folder** (macOS Sonoma and later), or open Terminal and type `cd ` then drag the folder into the window and press Enter.
- **Linux** — right-click inside the folder in your file manager and choose **Open Terminal Here**, or use `cd` to navigate to it.

---

## Step 4 — Run the model

```
python bigram_model.py pride_and_prejudice_p1-5.txt
```

> On macOS/Linux you may need to use `python3` instead of `python`.

You will see output like:

```
Reading 'pride_and_prejudice_p1-5.txt' ...
  Total tokens   : 1,824
  Vocabulary size: 482  (including EOS)
  Bigram table built.

Max tokens to generate (Enter = EOS only, 'q' to quit):
```

---

## Step 5 — Use the model

**Set a token limit**

- Press **Enter** to run in EOS-only mode — generation stops automatically at the end of a sentence.
- Or type a number (e.g. `20`) and press Enter to cap output at that many tokens.

**Enter a prompt**

Type one or more words that appear in the text. The model is case-insensitive, so `Bennet`, `bennet`, and `BENNET` are all the same.

```
Prompt [limit=EOS only] ('q' to quit): mr bennet
```

Example output:

```
------------------------------------------------------------
PROMPT   : mr bennet
GENERATED: was so odd a mixture of quick parts sarcastic humour reserve and caprice <EOS>
FULL TEXT: mr bennet was so odd a mixture of quick parts sarcastic humour reserve and caprice.
------------------------------------------------------------
```

**Change the token limit mid-session**

Press **F1** at the prompt to be asked for a new limit without restarting.

**Quit**

Type `q` and press Enter at any prompt.

---

## Using your own text file

You can use any plain `.txt` file instead of the provided one:

```
python bigram_model.py my_book.txt
```

The larger and more varied the text, the more interesting the generated output will be. The model works best on prose with normal sentence punctuation.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: command not found` | Try `python3` instead, or re-install Python and make sure to tick "Add to PATH" |
| `No such file or directory` | Make sure your terminal is in the same folder as `bigram_model.py` (see Step 3) |
| `Unknown token(s): [...]` | The word you typed is not in the training text — try a different word from the hint shown |
| Nothing generated, stops immediately | The last word in your prompt is always followed by a sentence end in the text — try a different prompt |
