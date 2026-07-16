
from nv_one_logger._dummy import _Dummy
def __getattr__(name): return _Dummy
