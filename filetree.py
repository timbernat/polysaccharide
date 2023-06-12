import json
from pathlib import Path
from subprocess import Popen

from typing import Any, Callable, Optional, Union
from abc import ABC, abstractstaticmethod

from .general import iter_len

class FileTypeError(Exception):
    '''Raise when file extension is not valid for a particular application'''
    pass


# FILE I/O functions
startfile = lambda path : Popen(['xdg-open', path]) # Replacement for os.startfile() functionality, since none natively exists in Linux

def filter_txt_by_condition(in_txt_path : Path, condition : Callable[[str], bool], out_txt_path : Optional[Path]=None, postfix : str='filtered', inclusive : bool=True, return_filtered_path : bool=False) -> Optional[Path]:
    '''Create a copy of a text-based file containing only the lines which match to a given boolean condition
    
    If no explicit output path is given, will create an output file in the same directory as the source file
    with the same name plus "postfix" tacked on. Can optionally return the path to the filtered file (else None)

    "Inclusive" kw governs whether to write lines which DO or DON'T meet the condition'''
    if out_txt_path is None:
        out_txt_path = in_txt_path.with_stem(f'{in_txt_path.stem}{"_" if postfix else ""}{postfix}')

    if (out_txt_path == in_txt_path):
        raise PermissionError(f'Attempting to overwrite {in_txt_path} with regex filter') # prevent write clash
    
    if (out_txt_path.suffix != in_txt_path.suffix):  # prevent file type conversion during transfer
        raise FileTypeError(f'Input and output file must have same extension (not {in_txt_path.suffix} and {out_txt_path.suffix})')

    with out_txt_path.open('w') as outfile: 
        with in_txt_path.open('r') as infile: # readfile is innermost in case error occurs during file read (caught by handler one level up)
            for line in infile:
                if (condition(line) == inclusive): # only write lines if (matching AND inclusive) OR (not matching AND exclusive)
                    outfile.write(line)

    if return_filtered_path:
        return out_txt_path
    
# pathlib Path-manipulation functions
dotless = lambda suffix : suffix.split('.')[-1] # separate the dot from a SINGLE extension file suffix

def is_empty(path : Path) -> bool:
    '''Check if a directory is empty'''
    assert(path.is_dir())
    return iter_len(path.iterdir()) == 0 # can't use "len" for generators :(

def clear_dir(path : Path) -> None:
    '''Recursively clear contents of a directory at the given path (depth-first)'''
    assert(path.is_dir())

    for sub_path in path.iterdir():
        if sub_path.is_dir():
            clear_dir(sub_path)
            sub_path.rmdir() # raises OSError if inside of target subfolder while being deleted
        else:
            sub_path.unlink()

def default_suffix(path : Path, suffix : str) -> Path:
    '''Asserts that a path has a suffix, appending a specified default suffix if none exists'''
    if not path.suffix:
        path = path.with_name(f'{path.stem}.{suffix}') # ensure charge params path has correct extension

    return path

def prepend_parent(path : Path, new_parent : Path) -> Path:
    '''Prepends a parent tree to an existing path'''
    return new_parent / path

def detach_parent(path : Path, old_parent : Path) -> Path:
    '''Cuts off a parent tree from an existing path'''
    return path.relative_to(old_parent)

def exchange_parent(path : Path, old_parent : Path, new_parent : Path) -> Path:
    '''Exchanges the parent tree of a path for another parent tree'''
    return prepend_parent(path=detach_parent(path, old_parent), new_parent=new_parent)

def local_rename(path : Path, new_name : str) -> Path:
    '''Performs file rename relative to the parent directory (NOT the cwd)'''
    return path.rename(path.with_name(new_name))

def local_restem(path : Path, new_stem : str) -> Path:
    '''Performs file rename relative to the parent directory (NOT the cwd), preserving the extension of the original file'''
    return path.rename(path.with_stem(new_stem))

# JSON-specific functionality
JSONSerializable = Union[str, bool, int, float, tuple, list, dict] 

def append_to_json(json_path : Path, **kwargs) -> None:
    '''Add an entry to an existing JSON file'''
    with json_path.open('r') as json_file:
        jdat = json.load(json_file)

    jdat.update(**kwargs)

    with json_path.open('w') as json_file:
        jdat = json.checkpoint(jdat, json_file, indent=4)

class JSONifiable(ABC):
    '''Base class which allows a child class to have its attributes written to and from a JSON file on-disc between interpreter sessions
    Children must implement how dict data (i.e. self.__dict__) is encoded to and decoded from JSON formatted dict'''

    # JSON encoding and decoding
    @abstractstaticmethod
    def serialize_json_dict(unser_jdict : dict[Any, Any]) -> dict[str, JSONSerializable]:
        '''For converting selfs __dict__ data into a form that can be serialized to JSON'''
        pass
    
    @abstractstaticmethod
    def unserialize_json_dict(ser_jdict : dict[str, JSONSerializable]) -> dict[Any, Any]:
        '''For de-serializing JSON-compatible data into a form that the __init__method can accept'''
        pass

    # File I/O
    def to_file(self, savepath : Path) -> None:
        '''Store parameters in a JSON file on disc'''
        assert(savepath.suffix == '.json')
        with savepath.open('w') as dumpfile:
            json.dump(self.__class__.serialize_json_dict(self.__dict__), dumpfile, indent=4)

    @classmethod
    def from_file(cls, loadpath : Path) -> 'JSONifiable':
        assert(loadpath.suffix == '.json')
        with loadpath.open('r') as loadfile:
            params = json.load(loadfile, object_hook=cls.unserialize_json_dict)

        return cls(**params)
    
    @staticmethod
    def update_checkpoint(funct : Callable) -> Callable[[Any], Optional[Any]]: # NOTE : this deliberately doesn't have a "self" arg!
        '''Decorator for updating the on-disc checkpoint file after a function updates a Polymer attribute'''
        def update_fn(self, *args, **kwargs) -> Optional[Any]:
            ret_val = funct(self, *args, **kwargs) # need temporary value so update call can be made before returning
            self.to_file()
            return ret_val
        return update_fn

class JSONDict(dict):
    '''Dict subclass which also updates an underlying JSON file - effectively and on-disc dict
    !NOTE! - JSON doesn't support non-string keys, so all keys given will be stringified - plan accordingly!'''
    def __init__(self, json_path : Path, *args, **kwargs):
        if isinstance(json_path, str):
            json_path = Path(json_path) # make input arg a bit more flexible to str input from user end

        if json_path.suffix != '.json':
            raise ValueError(f'The path "{json_path}" does not point to a .json file')
        
        self.json_path : Path = json_path
        if self.json_path.exists():
            try:
                kwargs.update(self._read_file(json_path))
            except json.JSONDecodeError: # catches Paths which point to incorrectly formatted JSONs - TODO: revise terrible except-pass structure
                pass
        else:
            self.json_path.touch()

        super().__init__(*args, **kwargs)
        self._update_file() # ensure file contains current contents post-init

    @staticmethod
    def _read_file(json_path : Path) -> dict:
        with json_path.open('r') as file:
            return json.load(file)        

    def _update_file(self, indent : int=4):
        '''Save current dict contents to JSON file'''
        with self.json_path.open('w') as file:
            json.dump(self, file, indent=indent)

    def __setitem__(self, __key: str, __value: JSONSerializable) -> None:
        super().__setitem__(__key, __value)
        self._update_file()

    def __delitem__(self, __key: str) -> None:
        super().__delitem__(__key)
        self._update_file()
