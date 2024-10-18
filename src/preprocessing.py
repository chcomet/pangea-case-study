import copy
import json
import os
import re

import magic_pdf.model as model_config

from tqdm import tqdm
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from typing import Set, Tuple, Any

model_config.__use_inside_model__ = True


def pdf_parser(pdf_name: str, bronze_path: str = "data/bronze", silver_path: str = "data/silver") -> None:
    # config paths
    pdf_path = os.path.join(bronze_path, f"{pdf_name}.pdf")
    output_path = os.path.join(silver_path, pdf_name)
    output_image_path = os.path.join(output_path, 'images')
    image_path_parent = os.path.basename(output_image_path)

    # read pdf file as bytes
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # using built-in model
    model_json = []

    # config writers
    image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

    # config pipe
    pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": model_json}, image_writer)

    # run pipe
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

    # update model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )

    # save intermediate data
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )

    # save markdown file
    md_writer.write(
        content=md_content,
        path=f"{pdf_name}.md"
    )


def _find_table_image_paths(data: Any, paths: Set[Tuple[str, str]] = None) -> Set[Tuple[str, str]]:
    if paths is None:
        paths = set()

    if isinstance(data, dict):
        # if current data is a dictionary, check if it contains 'image_path' key
        image_path = data.get("image_path")
        item_type = data.get("type")
        # when item_type is 'table' and image_path is not None, add to paths
        if item_type == "table" and image_path:
            paths.add((item_type, image_path))

        # recursively check all key-value pairs in the dictionary
        for key, value in data.items():
            _find_table_image_paths(value, paths)

    elif isinstance(data, list):
        # if current data is a list, recursively check all items in the list
        for item in data:
            _find_table_image_paths(item, paths)

    return paths


def replace_markdown_tables(paper_name: str) -> str:
    base_path = os.path.join('data/silver', paper_name)
    markdown_file_path = os.path.join(base_path, "auto", f"{paper_name}.md")
    json_file_path = os.path.join(base_path, "auto", f"{paper_name}_middle.json")

    # check if all files exist
    if any([not os.path.exists(path) for path in [base_path, markdown_file_path, json_file_path]]):
        raise FileNotFoundError(f"Incomplete files for {paper_name}")

    # read files
    with open(markdown_file_path, 'r', encoding='utf-8') as markdown_file:
        markdown_content = markdown_file.read()
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # find all image paths for tables
    image_paths = _find_table_image_paths(json_data)

    # process table images with OCR
    table_engine = RapidTable()
    ocr_engine = RapidOCR()
    for item_type, img_path in tqdm(image_paths):
        # config image path
        img_path = f"images/{img_path}"
        img_full_path = os.path.join(base_path, 'auto', img_path)

        # check if image file exists
        if not os.path.exists(img_full_path):
            raise FileNotFoundError(f"Image file not found: {img_full_path}")

        # perform OCR
        ocr_result, _ = ocr_engine(img_full_path)
        table_html_str, _, _ = table_engine(img_full_path, ocr_result)

        # replace image with table HTML in markdown content
        image_pattern = f"!\\[\\]\\({re.escape(img_path)}\\)"
        markdown_content = re.sub(image_pattern, table_html_str, markdown_content)

    return markdown_content


def select_related_sections(markdown_content: str) -> str:
    # define irrelevant sections
    delete_sections = ["Abstract", "Keywords", "Acknowledgements", "References", "Appendix", "Supplementary Materials",
                       "Supporting Information", "Author's Contribution", "Conflict of Interest", "conclusion",
                       "conclusions", "data availability", "Article Information", "Article Infos:", "Experimental"]
    delete_sections = [section.lower() for section in delete_sections]

    lines = markdown_content.splitlines()
    relevant_lines = []

    skip_section = False
    for line in lines:
        # check if this line is a H1 header
        if re.match(r'^# ', line):
            # extract header content
            header = line[2:].strip().lower()

            # skip irrelevant sections
            if header in delete_sections:
                skip_section = True
                continue
            else:
                # reset flag and add header to output
                skip_section = False
                relevant_lines.append(line)

        # add line if it is in relevant section
        elif not skip_section:
            relevant_lines.append(line)

    return '\n'.join(relevant_lines)


def clean_markdown(markdown_content: str) -> str:

    # remove all URLs
    markdown_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown_content)

    # remove all images
    markdown_content = re.sub(r'!\[.*\]\(.*\)', '', markdown_content)

    # remove all code blocks
    markdown_content = re.sub(r'```[^`]+```', '', markdown_content)

    # remove all math expressions
    markdown_content = re.sub(r'\$[^$]+\$', '', markdown_content)

    # remove multiple spaces
    markdown_content = re.sub(r' {2,}', ' ', markdown_content)

    return markdown_content


def write_to_markdown_file(content: str, paper_name: str, gold_path: str = "data/gold") -> None:
    file_path = os.path.join(gold_path, f"{paper_name}.md")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
