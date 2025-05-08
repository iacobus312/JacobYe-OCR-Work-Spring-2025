# import libraries
import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import sys

# set `<your-endpoint>` and `<your-key>` variables with the values from the Azure portal
endpoint = "https://'insert user name here'.cognitiveservices.azure.com/"
key = "Insert API Key"

# Add the missing format_polygon function
def format_polygon(polygon):
    if not polygon:
        return "N/A"
    return ", ".join(["[{}, {}]".format(p.x, p.y) for p in polygon])

def analyze_layout(output_file=None):
    # If output_file is provided, redirect stdout to that file
    original_stdout = sys.stdout
    if output_file:
        sys.stdout = open(output_file, 'w', encoding='utf-8')
    
    try:
        # sample document
        formUrl = "https://raw.githubusercontent.com/iacobus312/yamitext/main/index.pdf"

        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        poller = document_analysis_client.begin_analyze_document_from_url(
            "prebuilt-layout", formUrl
        )
        result = poller.result()

        for idx, style in enumerate(result.styles):
            print(
                "Document contains {} content".format(
                    "handwritten" if style.is_handwritten else "no handwritten"
                )
            )

        for page in result.pages:
            print("----Analyzing layout from page #{}----".format(page.page_number))
            print(
                "Page has width: {} and height: {}, measured with unit: {}".format(
                    page.width, page.height, page.unit
                )
            )

            for line_idx, line in enumerate(page.lines):
                words = line.get_words()
                print(
                    "...Line # {} has word count {} and text '{}' within bounding polygon '{}'".format(
                        line_idx,
                        len(words),
                        line.content,
                        format_polygon(line.polygon),
                    )
                )

                for word in words:
                    print(
                        "......Word '{}' has a confidence of {}".format(
                            word.content, word.confidence
                        )
                    )

            for selection_mark in page.selection_marks:
                print(
                    "...Selection mark is '{}' within bounding polygon '{}' and has a confidence of {}".format(
                        selection_mark.state,
                        format_polygon(selection_mark.polygon),
                        selection_mark.confidence,
                    )
                )

        for table_idx, table in enumerate(result.tables):
            print(
                "Table # {} has {} rows and {} columns".format(
                    table_idx, table.row_count, table.column_count
                )
            )
            for region in table.bounding_regions:
                print(
                    "Table # {} location on page: {} is {}".format(
                        table_idx,
                        region.page_number,
                        format_polygon(region.polygon),
                    )
                )
            for cell in table.cells:
                print(
                    "...Cell[{}][{}] has content '{}'".format(
                        cell.row_index,
                        cell.column_index,
                        cell.content,
                    )
                )
                for region in cell.bounding_regions:
                    print(
                        "...content on page {} is within bounding polygon '{}'".format(
                            region.page_number,
                            format_polygon(region.polygon),
                        )
                    )

        print("----------------------------------------")
    
    finally:
        # Restore stdout even if an error occurs
        if output_file:
            sys.stdout.close()
            sys.stdout = original_stdout


if __name__ == "__main__":
    # Specify the output file path where you want to save the results
    output_file_path = 'insert path here'
    
    # Call the function with the output file path
    analyze_layout(output_file_path)
    
    # Print confirmation message to console
    print(f"Analysis results have been saved to: {output_file_path}")