class _Dummy:
    def __init__(self,*a,**k): pass
    def __call__(self,*a,**k): return self
    def __getattr__(self,n): return _Dummy()
    def __iter__(self): return iter(())
    @classmethod
    def __class_getitem__(cls,item): return cls
