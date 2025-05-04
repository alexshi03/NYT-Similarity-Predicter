import csv
from pathlib import Path

from regex import process


columnNames = ["Author", "Source", "NYT", "Genre", "PubDate", "Article Title", "Article Text", "URL"]

if __name__ == "__main__":
    data = []
    pdfPaths = list(Path("./texts").rglob("*.pdf"))

    print("Beginning processing " + str(len(pdfPaths)) + " files")
    for idx, path in enumerate(pdfPaths):
        if idx % 10 == 0:
            print(str(idx+1)+"/"+str(len(pdfPaths)))
        pdfDict = process(path.as_posix())
        data.append(pdfDict)
    print("Finished processing " + str(len(data)) + " files")

    with open("./outputs/output_full.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=columnNames)
        writer.writeheader()
        writer.writerows(data)
    print("Finished writing")
