from langchain_community.document_loaders import CSVLoader, DataFrameLoader
import pandas as pd
def load_documents(file_path: str):
    df = pd.read_csv(
        file_path,
    )
    df = df.dropna().reset_index(drop=True)
    loader = DataFrameLoader(
        df,
        page_content_column='content'
    )

    docs = loader.load()
    return docs