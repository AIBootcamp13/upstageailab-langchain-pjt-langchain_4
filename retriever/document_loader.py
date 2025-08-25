from langchain_community.document_loaders import CSVLoader, DataFrameLoader
import pandas as pd
def load_documents(file_path: str):
    df = pd.read_csv(
        file_path,
        usecols=['title', 'content', 'date'] # 사용할 컬럼 명시적으로 지정
    )
    loader = DataFrameLoader(
        df,
        page_content_column='content'
    )

    docs = loader.load()
    return docs