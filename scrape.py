import re
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import csv
import time


def deversioner(versioned):
    match = re.match(r"(\d+\.\d+)(v\d+)?", versioned)
    if match:
        versioned_part = match.group(2)
        if versioned_part:
            versioned = versioned.replace(versioned_part, "")
        return match.group(1)
    return None


def get_field(subject):
    subject_fields = {
        "Physics": [
            "astro-ph",
            "cond-mat",
            "gr-qc",
            "hep-ex",
            "hep-lat",
            "hep-ph",
            "hep-th",
            "math-ph",
            "nlin",
            "nucl-ex",
            "nucl-th",
            "nucl-th",
            "quant-ph",
        ],
        "Mathematics": ["math"],
        "Computer Science": ["cs"],
        "Quantitative Biology": ["q-bio"],
        "Quantitative Finance": ["q-fin"],
        "Statistics": ["stat"],
        "Electrical Engineering and Systems Science": ["eess"],
        "Economics": ["econ"],
    }

    for field, subjects in subject_fields.items():
        if subject in subjects:
            return field

    return "Other"


def scrape_arxiv_data(subjects):
    # Open a CSV file for writing
    csv_file_path = "daily-arxiv-embeddings.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        # Include all fields in the CSV
        fieldnames = [
            "arxiv",
            "field",
            "subject",
            "categories",
            "authors",
            "title",
            "abstract",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for subject in subjects:
            url = f"https://arxiv.org/list/{subject}/new"
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            link_id = soup.select_one("#dlpage li:nth-child(2) a")["href"]
            id_match = re.search(r"#item(\d+)", link_id)
            id_number = int(id_match.group(1)) - 1 if id_match else 0

            links = soup.select(".list-identifier a:nth-child(1)")
            ids = [link["href"].split("/")[-1] for link in links[:id_number]]

            # Define the maximum number of IDs per request
            max_ids_per_request = 400
            for i in range(0, len(ids), max_ids_per_request):
                chunk_ids = ids[i : i + max_ids_per_request]
                params = {
                    "id_list": ",".join(chunk_ids),
                    "start": 0,
                    "max_results": 2000,
                }
                api_url = "http://export.arxiv.org/api/query"

                response = requests.get(api_url, params=params)
                if response.status_code != 200:
                    raise ConnectionError(
                        f"Failed to fetch data from the arXiv API for subject {subject}."
                    )

                root = ET.fromstring(response.content)

                namespaces = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom",
                }

                field = get_field(subject)

                # Iterate over entries and write to CSV
                for entry in root.findall("atom:entry", namespaces):
                    arxiv = deversioner(
                        entry.find("atom:id", namespaces).text.strip().split("/")[-1]
                    )
                    title = entry.find("atom:title", namespaces).text.strip()
                    categories_list = [
                        category.attrib["term"]
                        for category in entry.findall("atom:category", namespaces)
                    ]
                    authors_list = [
                        author.find("atom:name", namespaces).text.strip()
                        for author in entry.findall("atom:author", namespaces)
                    ]
                    abstract = entry.find("atom:summary", namespaces).text.strip()

                    # Directly write all data to the CSV file
                    csv_writer.writerow(
                        {
                            "arxiv": arxiv,
                            "field": field,
                            "subject": subject,
                            "categories": ",".join(categories_list),
                            "authors": ",".join(authors_list),
                            "title": title,
                            "abstract": abstract,
                        }
                    )

                print(f"Data for subject {subject} successfully added to the CSV file.")
                # Be gentle with arXiv
                time.sleep(3)

    print("Combined data for all subjects successfully saved to CSV file.")


# List of subjects to scrape
subjects_to_scrape = [
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "nucl-th",
    "quant-ph",
    "math",
    "cs",
    "q-bio",
    "q-fin",
    "stat",
    "eess",
    "econ",
]

# Call the scrape_arxiv_data function for all subjects
scrape_arxiv_data(subjects_to_scrape)
