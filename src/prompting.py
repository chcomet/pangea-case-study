import json
import os.path
import re

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter
from openai import OpenAI


def _construct_prompt(md_content: str) -> str:
    prompt = f"""
    Task 1: Identify all the binomial names of plants mentioned in the paper. 
    Task 2: Identify all the diseases and medical activities mentioned in the paper.
    Task 3: Identify all the chemicals compounds mentioned in the paper.
    Task 4: Identify all the locations e.g. Peru, South American, mentioned in the paper.
    Task 5: For each plant name identified in Task 1, determine if there is any mention of an associated disease or medical activity from Task 2 in the same text or surrounding context. If a relationship between the plant and the disease/medical activity is found, provide the evidence (specific text excerpt). Return the results in JSON format as a list of objects with the following structure:
        "disease": [
            {{
                "plant": The name of the plant,
                "disease/activity": The name of the associated disease or medical activity,
                "evidence": The text excerpt that supports the relation if true, or null if false
            }},
            ...
        ]
    Task 6: For each plant name identified in Task 1, determine if there is any mention of an associated chemical compound from Task 3 in the same text or surrounding context. If a relationship between the plant and the compound is found, provide the evidence (specific text excerpt). Return the results in JSON format as a list of objects with the following structure:
        "compound": [
            {{
                "plant": The name of the plant.
                "compound": The name of the associated chemical compound.
                "evidence": The text excerpt that supports the relation if true, or null if false.
            }},
            ...
        ]
    Task 7: For each plant name identified in Task 1, determine if there is any mention of an associated location from Task 4 in the same text or surrounding context. If a relationship between the plant and the compound is found, provide the evidence (specific text excerpt). Return the results in JSON format as a list of objects with the following structure:
        "location":[
            {{
                "plant": The name of the plant.
                "location": location mentioned in the paper.
                "evidence": The text excerpt that supports the relation if true, or null if false.
            }},
            ...
        ]

    Only return final output containing the results of Task 5,6,7 in the form of a JSON object as follows:
    {{"disease": [...], "compound": [...], "location": [...]}}

    Scientific Paper:
    <<<{md_content}>>>
    """
    return prompt


def _get_response(prompt: str, client: OpenAI, model="gpt-3.5-turbo") -> str:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert in ethnobotany. Identify the relations based on the scientific paper. Provide the final result as a JSON list without markdown format."},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=0,
    )
    return response.choices[0].message.content


def _direct_completion(markdown_content: str, client: OpenAI, model: str = "gpt-3.5-turbo") -> str:
    prompt = _construct_prompt(markdown_content)
    return _get_response(prompt, client, model)


def _get_chunked_completion(markdown_content: str, client: OpenAI, model: str = "gpt-3.5-turbo") -> str:
    headers_to_split_on = [("#", "Header 1")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    text_chunks = [chunk.page_content for chunk in markdown_splitter.split_text(markdown_content)]
    responses = []
    for chunk in text_chunks:
        responses.append(_direct_completion(chunk, client, model))
        print("CHUNK", len(responses))
        print(responses[-1])

    merged_data = {
        "disease": [],
        "compound": [],
        "location": []
    }
    for res in responses:
        data = json.loads(re.sub(r'```(JSON|json)([^`]+)```', r'\2', res))
        if isinstance(data, list):
            for item in data:
                merged_data["disease"].extend(item.get("disease", []))
                merged_data["compound"].extend(item.get("compound", []))
                merged_data["location"].extend(item.get("location", []))
        elif isinstance(data, dict):
            merged_data["disease"].extend(data.get("disease", []))
            merged_data["compound"].extend(data.get("compound", []))
            merged_data["location"].extend(data.get("location", []))

    return json.dumps(merged_data, indent=4)


def save_to_json_file(data: str, paper_name: str, result_path: str = "data/results") -> None:
    with open(os.path.join(result_path, f"{paper_name}.json"), "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(data, indent=4))


def extract_data(markdown_content: str, client: OpenAI, model: str = "gpt-3.5-turbo") -> str:
    prompt = _construct_prompt(markdown_content)
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    if len(encoding.encode(prompt)) > 4096:
        return _get_chunked_completion(markdown_content, client, model)
    else:
        return _direct_completion(markdown_content, client, model)


with open("data/gold/sample_paper_2.md", "r", encoding="utf-8") as paper_file:
    paper_content = paper_file.read()

res = extract_data(paper_content, OpenAI(api_key="sk-XdyxI3gxNDua2NHC5b1a2f6834734dF89542A9BeEbFfB342", base_url="https://free.gpt.ge/v1/"), "gpt-3.5-turbo")

print(res)