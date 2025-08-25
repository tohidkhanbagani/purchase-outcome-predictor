from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Literal
import pandas as pd
from src.utils.functions import ModelSolutions


class Ingest(BaseModel):
    folders: Optional[List[str]] = None
    file_name: Optional[str] = None

    def data_ingest(
        self,
        folders: Optional[List[str]] = None,
        file_name: Optional[str] = None
    ) -> pd.DataFrame:
        final_folders = folders or self.folders
        final_file_name = file_name or self.file_name

        if not final_folders or not final_file_name:
            raise ValueError("Both 'folders' and 'file_name' must be provided.")

        model = ModelSolutions()
        return model.load_data(final_folders, final_file_name)

    def data_validate(
        self,
        obj: Optional[pd.DataFrame] = None,
        folders: Optional[List[str]] = None,
        file_name: Optional[str] = None
    ) -> Dict:
        df = obj if obj is not None else self.data_ingest(folders, file_name)
        results = {}

        try:
            results["is_empty"] = df.empty
            results["shape"] = df.shape
            results["columns"] = df.columns.tolist()
            results["missing_values"] = df.isnull().sum().to_dict()
            results["duplicates"] = df.duplicated().sum()
            results["dtypes"] = df.dtypes.astype(str).to_dict()
            results["status"] = "Validation passed"
        except Exception as e:
            results["status"] = "Validation failed"
            results["error"] = str(e)

        return results

    def schema(
        self,
        choice: Literal["dataframe", "dictionary", "col_list"],
        obj: Optional[pd.DataFrame] = None,
        folders: Optional[List[str]] = None,
        file_name: Optional[str] = None
    ) -> Union[pd.DataFrame, Dict[str, str], List[str]]:
        df = obj if obj is not None else self.data_ingest(folders, file_name)

        if choice == "dataframe":
            return pd.DataFrame({
                "Column": df.columns,
                "Dtype": df.dtypes.astype(str).values
            })

        elif choice == "dictionary":
            return {col: str(dtype) for col, dtype in df.dtypes.items()}

        elif choice == "col_list":
            return df.columns.tolist()

        else:
            raise ValueError("choice must be one of: 'dataframe', 'dictionary', 'col_list'")
