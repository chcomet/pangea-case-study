# pangea-case-study

## Tasks
### Objective
Build a basic extraction pipeline to extract ethnobotanical data from scientific texts, focusing on plant-disease, plant-compound, and plant-location relationships, and evaluate on the pipeline performance.

### Task Detail
#### Source Material
Please find below 2 articles that we would like you to apply a LLM approach to retrieve data from. Feel free to expand or select similar additional articles if desired.
- [A study on different plants of Apocynaceae Family and their medicinal uses](https://ujpronline.com/index.php/journal/article/view/235)
- [Anti-inflammatory and analgesic components from "hierba santa," a traditional medicine in Peru](https://pubmed.ncbi.nlm.nih.gov/19067116/)

#### Data schema
Please extract the following fields:
- **Plant-Disease / Activity**: Identify which plants are used to treat or manage which specific diseases or activities.
  - Example:
    - Plant: Papaver Somniferum
    - Disease / activity: Analgesic
- **Plant-Compound**: Extract information about specific compoundsfound in these plants
    - Examples:
      - Plant: Papaver Somniferum
      - Compounds: Morphine, Codeine
- **Plant-Location of origin**: Determine the geographical locations where these plants are traditionally used.
  - Note: While not all ‘locations’ are written in a consistent fashion, for now, feel free to simplify by referencing relevant locations regardless of how it is structured.
  - Examples:
    - Plant: Papaver Somniferum
    - Location of Origin: China, Europe, Greek, Roman, etc
  
### Prompt Engineering for Data Extraction
We would like for you to build a basic extraction pipeline using GPT3.5 or any other free to use LLM of your choice. Your task is twofold.
1. Implement the extraction pipeline using prompting methods of your choice
   - What pre-processing steps would you take for the text sources?
   - Give reasons for the methods you have taken. Some options of prompting methods (but are not limited to):
     - Few shot prompting: add examples of the tasks
     - Prompt chaining: split tasks into subtasks
     - Verification prompt: verify the responses or information by cross-checking and substantiating it with additional reasoning or evidence
2. Evaluation of extraction pipeline
   - How would you evaluate the prompting methods? 
   - What performance metrics would you consider? 
   - Can you estimate the performance metrics using the information in the 2 articles?

### Additional guidelines
- Use the appropriate programming language and libraries of your choice.
- Make sure that the code is well-written, preferably in an object-oriented manner.

### Presentation of case study
1. Present your solutions and results: < 20 minutes

    Outline:
    - Scope of the case study
    - Approach taken
    - Results
    - Challenges and potential solutions
    - How you would think about scaling this to a diverse set of source material
2. Walk us through your code base: <10 minutes
   
    Feel free to share your code beforehand
3. Discussion and Q&A: ~30 minutes

## Implementation

### 1. Convert PDF to Markdown using MinerU 

```bash
docker build -t mineru:latest .
docker run -v G:\WS24\pangea-case-study\data:/data --rm -it --gpus=all mineru:latest /bin/bash
magic-pdf -p /data/bronze/sample_paper_1.pdf -o /data/silver/
```

### 2. Markdown pre-processing


### 3. Prompting 
OpenAI API source: https://github.com/popjane/free_chatgpt_api

```text
Task 1: Identify all the binomial names of plants mentioned in the paper. 
Task 2: Identify all the diseases and medical activities mentioned in the paper.
Task 3: Identify all the chemicals compounds mentioned in the paper.
Task 4: Identify all the locations e.g. Peru, South American, mentioned in the paper.
Task 5: For each plant name identified in Task 1, determine if there is any mention of an associated disease or medical activity from Task 2 in the same text or surrounding context. If a relationship between the plant and the disease/medical activity is found, provide the evidence (specific text excerpt). Return the results in JSON format as a list of objects with the following structure:
    "disease": [
        {{
            "plant": The name of the plant,
            "disease/activity": The name of the associated disease or medical activity,
            "evidence": Text excerpt that supports the relation if true, or null if false
        }},
        ...
    ]
Task 6: For each plant name identified in Task 1, determine if there is any mention of an associated chemical compound from Task 3 in the same text or surrounding context. If a relationship between the plant and the compound is found, provide the evidence (specific text excerpt). Return the results in JSON format as a list of objects with the following structure:
    "compound": [
        {{
            "plant": Name of the plant.
            "compound": Name of the associated chemical compound.
            "evidence": Text excerpt that supports the relation if true, or null if false.
        }},
        ...
    ]
Task 7: For each plant name identified in Task 1, determine if there is any mention of an associated location from Task 4 in the same text or surrounding context. If a relationship between the plant and the compound is found, provide the evidence (specific text excerpt). Return the results in JSON format as a list of objects with the following structure:
    "location":[
        {{
            "plant": Name of the plant.
            "location": Location mentioned in the paper.
            "evidence": Text excerpt that supports the relation if true, or null if false.
        }},
        ...
    ]

Only return final output containing the results of Task 5,6,7 in the form of a JSON object as follows:
{{"disease": [...], "compound": [...], "location": [...]}}

Scientific Paper:
<<<{md_content}
```