import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Sequence
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
import csv
from io import TextIOWrapper
from langchain.document_loaders.helpers import detect_file_encodings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NougatPDFLoader(BasePDFLoader):
    """A PDF loader class for converting PDF files to MMD.

    This class leverages the `nougat` library to perform the conversion from PDF to HTML.
    It inherits from `BasePDFLoader` and extends its functionality to utilize the `nougat` library.
    TODO, the load_and_split method need to be implemented and default is RecursiveCharacterTextSplitter
    Attributes:
        file_path (str): The path to the PDF file to be loaded.
        headers (Optional[Dict]): Optional headers to be used when loading the PDF.

    Raises:
        ImportError: If the `nougat` library is not installed.
        RuntimeError: If the `nougat` command fails to execute successfully.
    """

    def __init__(self, file_path: str, *, headers: Optional[Dict] = None):
        """Initialize with a file path."""
        try:
            import nougat
        except ImportError:
            raise ImportError(
                "Please install nougat to use NougatPDFLoader. "
                "You can install it with `pip install nougat`."
            )

        super().__init__(file_path, headers=headers)

    def nougat(self, file_path: Path) -> str:
        """Executes the `nougat` command to convert the specified PDF file to Markdown format.

        Args:
            file_path (Path): The path to the PDF file to be converted.

        Returns:
            str: The Markdown content resulting from the `nougat` conversion.
        """
        # nougat ./paperSnapshot.pdf --full-precision --markdown -m 0.1.0-base -o tmp --recompute
        cli_command = ["nougat", str(file_path), "full-precision", "--markdown", "-m", "0.1.0-base", "-o", "tmp", "--recompute"]

        try:
            result = subprocess.run(
                cli_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            result.check_returncode()
            return result.stdout

        except subprocess.CalledProcessError as e:
            logging.error(
                f"Nougat command failed with return code {e.returncode}: {e.stderr}"
            )
            raise RuntimeError("Nougat command failed.") from e

    def load(self) -> List[Document]:
        """Loads and processes the specified PDF file, converting it to a list of Document objects.

        Returns:
            List[Document]: A list containing a single Document object with the processed content.
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load and process the specified PDF file, yielding Document objects.

        This method reads the PDF file, processes it using the `nougat` command,
        reads the resulting Markdown content, and yields a Document object with the content.
        """
        try:
            file_path = self.file_path
            # Call the method to run the Nougat OCR command
            self.nougat(file_path)

            # Rest of your code for reading and processing the output
            file_path = Path(file_path)
            output_path = Path("tmp") / f"{file_path.stem}.mmd"
            with output_path.open("r") as f:
                content = f.read()
            # consider math expressions are enclosed in \( and \) in Markdown
            content = (
                content.replace(r"\(", "$")
                .replace(r"\)", "$")
                .replace(r"\[", "$$")
                .replace(r"\]", "$$")
            )
            metadata = {"source": self.file_path}
            yield Document(page_content=content, metadata=metadata)

        except Exception as e:
            logging.error(f"An error occurred while processing the PDF: {str(e)}")


class CustomCSVLoader(CSVLoader):
    """Load a `CSV` file into a list of Documents.

    Each document represents one row of the CSV file. The rows are converted into markdown format based on row_count.

    Output Example:
        when row_count = 1, 
        page_document_1 contains:
        |index|name|
        |-|-|
        |1|Demo1|
        page_document_2 contains:
        |index|name|
        |-|-|
        |2|Demo2|

        when row_count = 3, 
        page_document_1 contains:
        |index|name|
        |-|-|
        |1|Demo1|
        |2|Demo2|
        |3|Demo3|
        page_document_2 contains:
        |index|name|
        |-|-|
        |4|Demo4|
        |5|Demo5|
        |6|Demo6|
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Sequence[str] = (),
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
        row_count: int = 1
    ):
        """

        Args:
            file_path: The path to the CSV file.
            source_column: The name of the column in the CSV file to use as the source.
              Optional. Defaults to None.
            metadata_columns: A sequence of column names to use as metadata. Optional.
            csv_args: A dictionary of arguments to pass to the csv.DictReader.
              Optional. Defaults to None.
            encoding: The encoding of the CSV file. Optional. Defaults to None.
            autodetect_encoding: Whether to try to autodetect the file encoding.
            row_count: How many row in a page document.
        """
        self.row_number = row_count
        super().__init__(file_path, source_column, metadata_columns,
                         csv_args, encoding, autodetect_encoding)

    def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
        docs = []

        csv_reader = csv.DictReader(csvfile, **self.csv_args)
        counter = 0
        for i, row in enumerate(csv_reader):
            # print(f"i: {i}")
            # print(f"row: {row}")
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else self.file_path
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )
            counter += 1

            if counter % self.row_number == 1:
                # First row with header and separator
                header = "|"
                md_separator = "|"
                row_content = "|"
                for k, v in row.items():
                    header += k + "|"
                    md_separator += "-|"
                    row_content += v + "|"
                row_content += "\n"
            elif counter % self.row_number == 0:
                if 1 == self.row_number:
                    header = "|"
                    md_separator = "|"
                    row_content = "|"
                    for k, v in row.items():
                        header += k + "|"
                        md_separator += "-|"
                        row_content += v + "|"
                else:
                    for k, v in row.items():
                        row_content += v + "|"
                content = header + "\n" + md_separator + "\n" + row_content
                print(f"markdown content: {content}")

                metadata = {"source": source, "row": i}
                for col in self.metadata_columns:
                    try:
                        metadata[col] = row[col]
                    except KeyError:
                        raise ValueError(
                            f"Metadata column '{col}' not found in CSV file.")
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
                counter = 0
            else:
                for k, v in row.items():
                    row_content += v + "|"
                row_content += "\n"

        return docs

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                docs = self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            docs = self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return docs


# local debugging purpose
# if __name__ == "__main__":
#     # local pdf file in current folder
#     loader = NougatPDFLoader('paperSnapshot.pdf')
#     data = loader.load()
#     logging.info("text: %s", data)


# TODO: Local debug CSV loader, remove it before release
# if __name__ == "__main__":
#     import uuid
#     import boto3
#     from datetime import datetime

#     s3 = boto3.client('s3')
#     now = datetime.now()
#     timestamp_str = now.strftime("%Y%m%d%H%M%S")
#     print(timestamp_str)
#     random_uuid = str(uuid.uuid4())[:8]
#     print(random_uuid)

#     def process_csv(csv_content: str, kwargs):
#         bucket_name = kwargs['bucket']
#         key = kwargs['key']
#         local_path = f'<path>/temp-{timestamp_str}-{random_uuid}.csv'
#         s3.download_file(bucket_name, key, local_path)

#         # loader = CustomCSVLoader(file_path=local_path, row_count=1)
#         # loader = CustomCSVLoader(file_path=local_path, row_count=999)
#         loader = CustomCSVLoader(file_path=local_path, row_count=2)
#         # loader = CustomCSVLoader(file_path=local_path, row_count=3)
#         data = loader.load()
#         # print(data)

#     # TSV
#     # process_csv("x", {'bucket': '<bucket_name>', 'key': 'athena_results/OrderTable.tsv'})
#     # CSV
#     process_csv("x", {'bucket': '<bucket_name>', 'key': 'athena_results/sdps-api-test-s3-key-58h54muj.csv'})
