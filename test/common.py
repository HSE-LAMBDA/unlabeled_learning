def import_lib():
  try:
    import psgd
    return psgd
  except ImportError:
    import sys
    import os.path as osp
    root = osp.join(
      osp.dirname(osp.dirname(__file__)),
      'psgd'
    )

    sys.path.insert(0, root)
