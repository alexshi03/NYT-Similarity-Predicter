import fitz
import re


def process(fileName: str, source=""):
    fileParts = fileName.split("/")

    if not source:
        if fileParts[1] == "NYT":
            source = "NYT"
        else:
            source = fileParts[2]

    if source == "NYT":
        return processNYT(fileName)
    elif source == "FinancialTimes":
        return processFT(fileName)
    elif source == "WashingtonPost":
        return processWP(fileName)


def processNYT(fileName: str):
    doc = fitz.open(filename=fileName)

    text = ""
    author = ""
    authors = []
    date = ""
    url = ""
    ret = {}

    title = doc.metadata.get("title")
    title = re.sub(" - The New York Times", "", title)

    for page_num, page in enumerate(doc):
        curr_text = page.get_text()
        text += curr_text

    # grab date
    match = re.search(
        r"[A-Z][a-z]+ \d{1,2}, \d{4}(?:, .*?)?\n", text, flags=re.MULTILINE
    )
    if match and date == "":
        date = match.group(0)

    # grab author(s)
    match = re.search(r"^By (.*)\n", text, flags=re.MULTILINE)
    if match and author == "":
        author = match.group(1)

    # grab url
    match = re.search(r"(?:https?://|www\.)[\S\s]+?\.html\b", text, flags=re.MULTILINE)
    if match and url == "":
        url = match.group(0)

    # remove new page
    pattern = r"""(?m)^           # start of line (multiline)
        \d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\n  # timestamp line
        .*\n                    # title line
        (?:https?://\S+|www\.\S+)\n  # URL line
        \d+/\d+\n?              # page count line
        """
    text = re.sub(pattern, "", text, flags=re.VERBOSE)

    # remove html header
    text = re.sub(
        r"(?:https?://|www\.)[\S\s]+?\.html\b\n", "", text, flags=re.MULTILINE
    )

    # remove header
    pattern = (
        r"((.*\n)*?Listen to this article.*\n)?"
        r"(?:.*\n)*?"  # lines in between
        r"^By .*\n"  # byline
        r"(?:.*\n)*?"  # optional description lines
        r"[A-Z][a-z]+ \d{1,2}, \d{4}(?:, .*?)?\n"  # date line
        r"(?:Updated .*\n)?"  # optional updated line
    )
    text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # remove title
    pattern = r"\s*".join(re.escape(word) for word in title.split()) + "\n"
    text = re.sub(pattern, "", text)

    # remove picture caption attributions
    pattern = r"^.*\sfor\sThe\sNew\sYork\sTimes\s*$"
    text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # remove "when we learn of a mistake"
    pattern = r"When we learn of a mistake, we acknowledge it with a correction. If you spot an error, please let us know at\nnytnews@nytimes.com. Learn more(\n)*"
    text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # remove author bios
    author = author.replace("\xa0", " ")
    authors = re.split(r", and | and |, ", author)
    text = removeAuthorBios(text, authors)

    # remove contribution lines
    pattern = r"^.*\bcontributed .*\.\s*$"
    match = None
    for m in re.finditer(pattern, text, flags=re.MULTILINE):
        match = m
    if match:
        end_pos = match.span()[1]
        if text[end_pos:].strip() == "":
            text = text[: match.start()].rstrip()

    doc.close()

    # set up return
    fileParts = fileName.split("/")

    ret["Author"] = ";".join(authors)
    ret["Source"] = fileParts[1]
    ret["NYT"] = "1" if fileParts[1] == "NYT" else "0"
    ret["Genre"] = fileParts[2]
    monthToNumber = {
        "January": "1",
        "February": "2",
        "March": "3",
        "April": "4",
        "May": "5",
        "June": "6",
        "July": "7",
        "August": "8",
        "September": "9",
        "October": "10",
        "November": "11",
        "December": "12",
    }
    dateParts = re.split(r"[ ,\n]+", date)
    ret["PubDate"] = (
        monthToNumber[dateParts[0]] + "/" + dateParts[1] + "/" + dateParts[2]
    )
    ret["Article Title"] = title
    ret["Article Text"] = text.strip()
    ret["URL"] = url.strip()

    return ret


def removeAuthorBios(text, author_names):
    lines = text.strip().splitlines()
    authors_seen = set()
    latest = 0

    for idx, line in enumerate(reversed(lines)):
        for name in author_names:
            if name in line:
                # print(idx, name, line)
                authors_seen.add(name)
                latest = idx + 1

        if len(authors_seen) == len(author_names):
            break

    remaining_lines = lines[: len(lines) - latest]
    return "\n".join(remaining_lines).strip()


def processFT(fileName):
    doc = fitz.open(filename=fileName)

    text = ""
    author = ""
    authors = []
    date = ""
    url = ""
    ret = {}

    title = doc.metadata.get("title")
    # title = re.sub(" - The New York Times", "", title)

    for page_num, page in enumerate(doc):
        curr_text = page.get_text()
        text += curr_text

    # grab date
    match = re.search(r"Published (.*)\n", text, flags=re.MULTILINE)
    if match and date == "":
        date = match.group(1)

    # grab author(s)
    match = re.search(r"(.*) in .*\n", text)
    if match and author == "":
        author = match.group(1)

    # grab url
    match = re.search(r"(?:https?://|www\.)[\S\s]+?.*\b", text, flags=re.MULTILINE)
    if match and url == "":
        url  = match.group(0)

    # remove title page
    title_pattern = r"\s*".join(re.escape(word) for word in title.split()) + r"\n"
    pattern = (
        r"^.*\n"
        + title_pattern
        + r"((?:.*\n)*?)"
        + r"\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\n"
    )
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match:
        text = re.sub(match.group(1), "", text, flags=re.MULTILINE, count=1)

    # remove new page
    pattern = r"""(?m)^           # start of line (multiline)
        \d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\n  # timestamp line
        .*\n                    # title line
        (?:https?://\S+|www\.\S+)\n  # URL line
        \d+/\d+\n?              # page count line
        """
    text = re.sub(pattern, "", text, flags=re.VERBOSE)

    # remove html header
    text = re.sub(
        r"(?:https?://|www\.)[\S\s]+?\.html\b\n", "", text, flags=re.MULTILINE
    )

    # remove header
    pattern = r"^(?:.*\n)*?^Published.*\n"
    text = re.sub(pattern, "", text, flags=re.MULTILINE, count=1)

    # remove title
    text = re.sub(title_pattern, "", text)

    # remove picture caption attributions
    pattern = r'^.*©.*$\n?'
    text = re.sub(pattern, '', text, flags=re.MULTILINE)

    pattern = r"Copyright The Financial Times Limited 2025\. All rights reserved\..*"
    text = re.sub(pattern, '', text, flags=re.DOTALL)

    # remove author bios
    author = author.replace('\xa0', ' ')
    authors = re.split(r", and | and |, ", author)

    doc.close()

    # set up return
    fileParts = fileName.split("/")

    ret["Author"] = ";".join(authors)
    ret["Source"] = fileParts[1]
    ret["NYT"] = "1" if fileParts[1] == "NYT" else "0"
    ret["Genre"] = fileParts[2]
    monthToNumber = {
        "january": "1",
        "february": "2",
        "march": "3",
        "april": "4",
        "may": "5",
        "june": "6",
        "july": "7",
        "august": "8",
        "september": "9",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    dateParts = re.split(r"[ ,\n]+", date)
    ret["PubDate"] = (
        monthToNumber.get(dateParts[0].lower(), "") + "/" + dateParts[1] + "/" + dateParts[2]
    )
    ret["Article Title"] = title
    ret["Article Text"] = text.strip()
    ret["URL"] = url.strip()

    return ret

def processWP(fileName: str):
    doc = fitz.open(filename=fileName)

    text = ""
    author = ""
    authors = []
    date = ""
    # url = ""
    ret = {}

    title = doc.metadata.get("title")
    title = re.sub(" - The Washington Post", "", title)

    for page_num, page in enumerate(doc):
        curr_text = page.get_text()
        text += curr_text

    # grab date
    match = re.search(
        r"[A-Z][a-z]+ \d{1,2}, \d{4}(?:, .*?)?\n", text, flags=re.MULTILINE
    )
    if match and date == "":
        date = match.group(0)

    # grab author(s)
    match = re.search(r"^By (.*)\n", text, flags=re.MULTILINE)
    if match and author == "":
        author = match.group(1)

    # grab url
    # no url

    # remove new page
    pattern = r"""(?m)^           # start of line (multiline)
        \d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s+[AP]M\n  # timestamp line
        .*\n                    # title line
        (?:https?://\S+|www\.\S+)\n  # URL line
        \d+/\d+\n?              # page count line
        """
    text = re.sub(pattern, "", text, flags=re.VERBOSE)


    # remove header
    pattern = (
        r"(?:.*\n)*?"  # lines in between
        r"[A-Z][a-z]+ \d{1,2}, \d{4}(?:, .*?)?\n"  # date line
        r"(?:Updated .*\n)?"  # optional updated line
        r"^By .*\n"  # byline
    )
    text = re.sub(pattern, "", text, flags=re.MULTILINE, count=1)

    # remove title
    pattern = r"\s*".join(re.escape(word) for word in title.split()) + "\n"
    text = re.sub(pattern, "", text)

    # remove picture caption attributions

    # remove "this summary is AI generated."
    pattern = r"What readers are saying.*"
    text = re.sub(pattern, "", text, flags=re.DOTALL)

    pattern = r"Trump presidency\nFollow live updates on the Trump administration. We’re tracking Trump’s\nprogress on campaign promises and legal challenges to his executive orders and\nactions.\n"
    text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # remove contribution lines
    pattern = r"^.*\bcontributed .*\.\s*$"
    match = None
    for m in re.finditer(pattern, text, flags=re.MULTILINE):
        match = m
    if match:
        end_pos = match.span()[1]
        if text[end_pos:].strip() == "":
            text = text[: match.start()].rstrip()

    # remove author bios
    author = author.replace('\xa0', ' ')
    authors = re.split(r", and | and |, ", author)

    # remove "Democracy Dies in Darkness"
    pattern = r"Democracy Dies in Darkness\n"
    text = re.sub(pattern, "", text)

    doc.close()

    # set up return
    fileParts = fileName.split("/")

    ret["Author"] = ";".join(authors)
    ret["Source"] = fileParts[1]
    ret["NYT"] = "1" if fileParts[1] == "NYT" else "0"
    ret["Genre"] = fileParts[2]
    monthToNumber = {
        "January": "1",
        "February": "2",
        "March": "3",
        "April": "4",
        "May": "5",
        "June": "6",
        "July": "7",
        "August": "8",
        "September": "9",
        "October": "10",
        "November": "11",
        "December": "12",
    }
    dateParts = re.split(r"[ ,\n]+", date)
    ret["PubDate"] = (
        monthToNumber.get(dateParts[0],"") + "/" + dateParts[1] + "/" + dateParts[2]
    )
    ret["Article Title"] = title
    ret["Article Text"] = text.strip()
    ret["URL"] = "NA"

    return ret