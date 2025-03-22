from fuse import *


def test_comparison():
    cell = Point(0)
    l2 = CellL2(cell)
    h1 = CellH1(cell)
    hdiv = CellHDiv(cell)
    hcurl = CellHCurl(cell)

    assert h1 < l2
    assert hdiv < l2
    assert hcurl < l2
    assert not h1 > hcurl
