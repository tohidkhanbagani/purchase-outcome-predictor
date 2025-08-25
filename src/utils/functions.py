import joblib
import pandas as pd
from pathlib import Path
from typing import Any,List
from src.utils.path_utils import get_project_root


class ModelSolutions:
       
       def __init__(self):
        self.project_root = get_project_root()
       
       
       def _path_resolver(self ,path:List[str],file_name:str)->Path:
              
              full_path = self.project_root.joinpath(*path)
              full_path.mkdir(parents=True, exist_ok=True)
              return full_path / file_name
        #       project_root = os.path.abspath(os.path.join(os.getcwd(),".."))
        #       data_path = os.path.join(project_root,*path)
        #       os.makedirs(data_path,exist_ok=True)
        #       final_path = os.path.join(data_path,file_name)
        #       return final_path
       
       def load_data(self, folders:List[str]=None,file_name:str=None)->pd.DataFrame:
               data_file = pd.read_csv(self._path_resolver(path=folders,file_name=file_name))
               return data_file
       
       def load_model(self,folders:List[str]=None,file_name:str=None) -> Any:
               final_path = self._path_resolver(path=folders,file_name=file_name)
               if not Path.exists(final_path):
                      raise FileNotFoundError(f"Model file not found at: {final_path}")
               return joblib.load(final_path)
       
       def save_model(self,folders:list[str]=None,obj:Any=None,file_name:str=None)->Any:
               final_path = self._path_resolver(path=folders,file_name=file_name)
               return joblib.dump(obj,final_path)
        
       def dump_data(self,folders:List[str]=None,file_name:str=None,obj:Any=None)->Any:
               final_path = self._path_resolver(path=folders,file_name=file_name)
               if not isinstance(obj,pd.DataFrame):
                      raise TypeError("The File is Not a CSV Type")
               else:
                     return obj.to_csv(final_path,index=False)
               