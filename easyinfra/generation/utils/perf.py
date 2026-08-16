import time
from functools import wraps
from easyinfra.utils import synchronize_device, show_rank_print
from easyinfra import envs
from enum import Enum
import copy

class TimingMode(Enum):
    NONE = 0,
    SUM = 1,
    EACH = 2,

def timing_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        synchronize_device()
        start = time.perf_counter()
        # run func
        output = func(self, *args, **kwargs)
        synchronize_device()
        elapse = time.perf_counter() - start
        # update time elapse
        self.update_timing_results(func, elapse)
        return output
    return wrapper

def class_with_timing(cls):
    env_class_timing_mode = envs.CLASS_TIMING_MODE
    if env_class_timing_mode in ("0", "none"):
        timing_mode = TimingMode.NONE
    elif env_class_timing_mode in ("1", "sum"):
        timing_mode = TimingMode.SUM
    elif env_class_timing_mode in ("2", "each"):
        timing_mode = TimingMode.EACH
    else:
        raise ValueError
    
    cls.timing_mode = timing_mode
    if cls.timing_mode == TimingMode.NONE:
        def _dummy(*args, **kwargs):
            pass
        cls.update_timing_results = _dummy
        cls.show_timing_results = _dummy
        cls.init_clear_timing_results = _dummy
        return cls
    
    # define update function
    def update_timing_results(self, func, elapse: float):
        key = func.__name__
        if timing_mode == TimingMode.SUM:
            self.timing_results[key] += elapse
            # show_rank_print(f"1: {key}:{round(self.timing_results[key], 6)}", 0)
        elif timing_method == TimingMode.EACH:
            self.timing_results[key].append(elapse)
        else:
            raise ValueError
    
    def show_timing_results(self, rank = 0):
        if self.timing_mode != TimingMode.NONE:
            show_rank_print(f"timming results of mode {self.timing_mode}:", rank)
            if self.timing_mode == TimingMode.SUM:
                for key,value in self.timing_results.items():            
                    show_rank_print(f"{key}: {round(value, 6)}", rank)
                show_rank_print(f"TOTAL: {round(sum(value for key,value in self.timing_results.items()), 6)}", rank)
            else:
                raise NotImplementedError            
        
    timing_results_init_item = 0.0 if timing_mode == TimingMode.SUM else []
    timing_results_initialization = {}
    for name, value in cls.__dict__.items():
        ## if it is a computation method
        if callable(value) and not name.startswith("_"):
            timing_results_initialization[name] = copy.deepcopy(timing_results_init_item)
            setattr(cls, name, timing_method(value))
    
    def init_clear_timing_results(self):
        self.timing_results = copy.deepcopy(timing_results_initialization)
        
    cls.update_timing_results = update_timing_results
    cls.show_timing_results = show_timing_results
    cls.init_clear_timing_results = init_clear_timing_results
    
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.init_clear_timing_results()

    cls.__init__ = new_init

    return cls