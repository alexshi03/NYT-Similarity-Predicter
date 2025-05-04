import newspaper

if __name__ == "__main__":
    article = newspaper.article("https://www.ft.com/content/5ab8bce3-e3d9-48b8-a3a3-d8a563447ca8")

    with open('./ouptuts/newspaper_output4.txt', 'w') as f:
        print("\n".join(line for line in article.text.splitlines() if line.strip()), file=f)