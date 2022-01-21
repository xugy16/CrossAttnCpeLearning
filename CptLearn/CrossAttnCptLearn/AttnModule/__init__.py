"""
The basic logic is that:
1. we have MHCA to implement our models.
2. based on MHCA:
    2.1 SelfAttn
    2.2 CrossAttn
3. Then we have encoder-decoder framework.

And this implementation doesn't depend on the Bert implementation.
"""
from .MHAttn import MHAttn
from .MHCA_EncDec import MHCA_EncDec
from .GatedElementSum import GatedElementSum
from .GatedElementSumProj import GatedElementSumProj