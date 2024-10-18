from langchain_openai import OpenAI

from src import preprocessing, prompting


def execute(paper_name: str):
    # prerocessing
    preprocessing.pdf_parser(paper_name)
    markdown_content = preprocessing.replace_markdown_tables(paper_name)
    markdown_content = preprocessing.select_related_sections(markdown_content)
    markdown_content = preprocessing.clean_markdown(markdown_content)
    preprocessing.write_to_markdown_file(markdown_content, paper_name)
    # prompting
    client = OpenAI()
    json_result = prompting.extract_data(markdown_content, client)
    prompting.save_to_json_file(json_result, paper_name)


if __name__ == "__main__":
    for paper_name in ["sample_paper_1", "sample_paper_2"]:
        execute(paper_name)
