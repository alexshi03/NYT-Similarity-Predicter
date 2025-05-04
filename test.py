from regex import process

if __name__ == "__main__":
    pdfDict = process("./texts/NotNYT/WashingtonPost/China retaliates against Trump’s tariffs, imposes 34 percent levy on U.S. goods - The Washington Post.pdf", source="WashingtonPost")

    for key in pdfDict:
        # if key == "Article Text":
        #     continue
        print(key + ": " + pdfDict[key])